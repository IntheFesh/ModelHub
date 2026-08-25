"""Unit tests for train/dpo.py.

Preference-tiering/pairing tests build `PredictionRecord`s directly
(exhaustive over every real `exec_code`/`comparison_result` combination
`classify_prediction` must handle). `sample_k_candidates` tests reuse
A4's real `run_one_sample` against a real SQLite fixture db, only the
model-generation step faked (`tests/unit/a4/fakes.FakeModelClient`) —
same discipline as A4's own runner tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.unit.a4.fakes import FakeModelClient

from modelhub.common.capability import CheckStatus
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.model_client import GenerationResult
from modelhub.eval.records import PredictionRecord
from modelhub.train.dpo import (
    DiscardReason,
    DpoTrainingConfig,
    PreferenceTier,
    assess_expected_dpo_benefit,
    build_dpo_config_kwargs,
    build_preference_dataset,
    build_preference_pairs_for_question,
    check_trl_available,
    classify_prediction,
    run_dpo_training,
    sample_k_candidates,
)


def _prediction(**overrides: object) -> PredictionRecord:
    defaults: dict[str, object] = {
        "sample_id": "q0",
        "db_id": "school",
        "difficulty": "simple",
        "predicted_sql": "SELECT 1",
        "finish_reason": "stop",
        "exec_code": ErrorCode.EXEC_OK,
        "comparison_result": ComparisonResult.EQUAL,
        "elapsed_s": 0.01,
    }
    defaults.update(overrides)
    return PredictionRecord.model_validate(defaults)


class TestClassifyPrediction:
    def test_exec_ok_and_equal_is_correct_tier(self) -> None:
        c = classify_prediction("q0", 0, _prediction())
        assert c.tier is PreferenceTier.EXEC_OK_CORRECT
        assert c.discard_reason is None

    def test_exec_ok_and_not_equal_is_incorrect_tier(self) -> None:
        c = classify_prediction("q0", 0, _prediction(comparison_result=ComparisonResult.NOT_EQUAL))
        assert c.tier is PreferenceTier.EXEC_OK_INCORRECT

    @pytest.mark.parametrize("exec_code", [ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT])
    def test_syntax_semantic_timeout_are_exec_failed_tier(self, exec_code: ErrorCode) -> None:
        c = classify_prediction("q0", 0, _prediction(exec_code=exec_code, comparison_result=None))
        assert c.tier is PreferenceTier.EXEC_FAILED

    def test_unsafe_statement_is_exec_failed_tier(self) -> None:
        c = classify_prediction(
            "q0", 0, _prediction(exec_code=ErrorCode.UNSAFE_STATEMENT, comparison_result=None)
        )
        assert c.tier is PreferenceTier.EXEC_FAILED

    def test_harness_error_is_discarded_not_tiered(self) -> None:
        c = classify_prediction(
            "q0",
            0,
            _prediction(exec_code=ErrorCode.HARNESS_DB_UNAVAILABLE, comparison_result=None),
        )
        assert c.tier is None
        assert c.discard_reason is DiscardReason.HARNESS_ERROR

    def test_harness_internal_is_discarded_not_tiered(self) -> None:
        c = classify_prediction(
            "q0", 0, _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        )
        assert c.discard_reason is DiscardReason.HARNESS_ERROR

    def test_output_truncated_is_discarded(self) -> None:
        c = classify_prediction(
            "q0",
            0,
            _prediction(
                finish_reason="length", exec_code=ErrorCode.OUTPUT_TRUNCATED, comparison_result=None
            ),
        )
        assert c.discard_reason is DiscardReason.OUTPUT_TRUNCATED

    def test_undecidable_is_discarded(self) -> None:
        c = classify_prediction(
            "q0",
            0,
            _prediction(
                comparison_result=ComparisonResult.UNDECIDABLE,
                undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
            ),
        )
        assert c.discard_reason is DiscardReason.UNDECIDABLE

    def test_unclassified_is_discarded_not_tiered_and_not_raised(self) -> None:
        # Regression test: ErrorCode.UNCLASSIFIED is a real value every
        # sqlexec backend's exception classifier can and does emit as its
        # fallback (backends.py's sqlite/duckdb/postgres classifiers all
        # return it for an unrecognized DB error) — classify_prediction
        # must not crash on real eval-pipeline output. It also must not be
        # silently folded into EXEC_FAILED (that would blame the model for
        # a possibly-system-side bug CLAUDE.md §2.3 explicitly warns
        # about), so it gets its own discard reason.
        c = classify_prediction(
            "q0", 0, _prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None)
        )
        assert c.tier is None
        assert c.discard_reason is DiscardReason.UNCLASSIFIED

    def test_genuinely_unhandled_exec_code_still_raises(self) -> None:
        # classify_prediction's whitelist must still hard-fail on a code it
        # was never taught about — proves the exhaustiveness guard itself
        # still works now that UNCLASSIFIED has a real branch.
        with pytest.raises(ValueError, match="unhandled exec_code"):
            classify_prediction(
                "q0", 0, _prediction(exec_code=ErrorCode.RUN_POLLUTED, comparison_result=None)
            )


class TestBuildPreferencePairsForQuestion:
    def test_cross_tier_pairs_only_no_same_tier_pairs(self) -> None:
        predictions = [
            _prediction(predicted_sql="correct1"),  # tier EXEC_OK_CORRECT
            _prediction(predicted_sql="correct2"),  # tier EXEC_OK_CORRECT (same tier)
            _prediction(
                predicted_sql="wrong1", comparison_result=ComparisonResult.NOT_EQUAL
            ),  # tier EXEC_OK_INCORRECT
            _prediction(
                predicted_sql="bad_sql", exec_code=ErrorCode.SYNTAX, comparison_result=None
            ),  # tier EXEC_FAILED
        ]
        result = build_preference_pairs_for_question("q0", predictions)
        assert len(result.classified) == 4
        # 2 tier1 x 1 tier2 + 2 tier1 x 1 tier3 + 1 tier2 x 1 tier3 = 5 pairs
        assert len(result.pairs) == 5
        for pair in result.pairs:
            assert pair.chosen_sql != pair.rejected_sql
        # never a tier1-vs-tier1 pair (correct1/correct2 never paired against each other)
        same_tier_pairs = [p for p in result.pairs if p.chosen_tier == p.rejected_tier]
        assert same_tier_pairs == []

    def test_discarded_candidates_never_appear_in_a_pair(self) -> None:
        predictions = [
            _prediction(predicted_sql="correct1"),
            _prediction(
                predicted_sql="harness_fail",
                exec_code=ErrorCode.HARNESS_INTERNAL,
                comparison_result=None,
            ),
        ]
        result = build_preference_pairs_for_question("q0", predictions)
        assert result.pairs == ()  # only one kept candidate, nothing to pair against

    def test_all_same_tier_produces_no_pairs(self) -> None:
        predictions = [_prediction(predicted_sql="a"), _prediction(predicted_sql="b")]
        result = build_preference_pairs_for_question("q0", predictions)
        assert result.pairs == ()


class TestBuildPreferenceDataset:
    def test_aggregates_across_questions(self) -> None:
        q0 = [
            _prediction(predicted_sql="c0"),
            _prediction(predicted_sql="w0", comparison_result=ComparisonResult.NOT_EQUAL),
        ]
        q1 = [
            _prediction(predicted_sql="c1"),
            _prediction(predicted_sql="w1", comparison_result=ComparisonResult.NOT_EQUAL),
        ]
        report = build_preference_dataset({"q0": q0, "q1": q1})
        assert report.pair_count == 2
        assert report.total_candidates == 4
        assert report.tier_counts[PreferenceTier.EXEC_OK_CORRECT.value] == 2
        assert report.tier_counts[PreferenceTier.EXEC_OK_INCORRECT.value] == 2

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_preference_dataset({})

    def test_exec_ok_correct_share(self) -> None:
        q0 = [_prediction(predicted_sql="c0"), _prediction(predicted_sql="c1")]
        q1 = [_prediction(predicted_sql="bad0", exec_code=ErrorCode.SYNTAX, comparison_result=None)]
        report = build_preference_dataset({"q0": q0, "q1": q1})
        assert report.exec_ok_correct_share == pytest.approx(2 / 3)

    def test_discard_counts_tracked(self) -> None:
        q0 = [
            _prediction(predicted_sql="c0"),
            _prediction(
                predicted_sql="harness",
                exec_code=ErrorCode.HARNESS_DB_UNAVAILABLE,
                comparison_result=None,
            ),
        ]
        report = build_preference_dataset({"q0": q0})
        assert report.discard_counts[DiscardReason.HARNESS_ERROR.value] == 1

    def test_unclassified_rate_within_threshold_builds_normally(self) -> None:
        # 1 unclassified out of 100 candidates == 1%, not > 1%, so this must
        # not raise (CLAUDE.md §2.2's threshold is strictly "> 1%").
        q0 = [_prediction(predicted_sql=f"c{i}") for i in range(99)] + [
            _prediction(
                predicted_sql="mystery", exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None
            )
        ]
        report = build_preference_dataset({"q0": q0})
        assert report.discard_counts[DiscardReason.UNCLASSIFIED.value] == 1
        assert report.unclassified_rate == pytest.approx(0.01)

    def test_unclassified_flood_aborts_dataset_build(self) -> None:
        # Meta-test-shaped: inject a guaranteed-over-threshold unclassified
        # rate and assert build_preference_dataset actually refuses to
        # build a dataset, rather than silently training on a mostly-
        # unexplained sandbox (CLAUDE.md §1.5 test_skip_is_not_pass sibling
        # — an unclassified flood must not silently pass as "just some
        # discards").
        q0 = [
            _prediction(
                predicted_sql=f"mystery{i}",
                exec_code=ErrorCode.UNCLASSIFIED,
                comparison_result=None,
            )
            for i in range(5)
        ] + [_prediction(predicted_sql="c0")]
        with pytest.raises(ValueError, match="unclassified_rate"):
            build_preference_dataset({"q0": q0})


class TestAssessExpectedDpoBenefit:
    def test_high_share_flags_small_expected_benefit(self) -> None:
        q0 = [_prediction(predicted_sql=f"c{i}") for i in range(9)] + [
            _prediction(predicted_sql="w0", comparison_result=ComparisonResult.NOT_EQUAL)
        ]
        report = build_preference_dataset({"q0": q0})
        message = assess_expected_dpo_benefit(report)
        assert "already >=" in message

    def test_low_share_flags_meaningful_room(self) -> None:
        q0 = [_prediction(predicted_sql="c0")] + [
            _prediction(predicted_sql=f"w{i}", comparison_result=ComparisonResult.NOT_EQUAL)
            for i in range(9)
        ]
        report = build_preference_dataset({"q0": q0})
        message = assess_expected_dpo_benefit(report)
        assert "meaningful room" in message


def _sample(**overrides: object) -> NormalizedSample:
    defaults: dict[str, object] = {
        "sample_id": "q0",
        "db_id": "school",
        "question": "how many students?",
        "evidence": None,
        "gold_sql": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
        "source": Source.BIRD,
        "split": Split.TRAIN,
        "dialect": Dialect.SQLITE,
    }
    defaults.update(overrides)
    return NormalizedSample.model_validate(defaults)


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO students VALUES (?, ?)", [(1, "ada"), (2, "grace")])
    conn.commit()
    conn.close()
    return db_root


class TestSampleKCandidates:
    def test_calls_generate_k_times_and_executes_each(self, tmp_path: Path) -> None:
        responses = iter(
            [
                GenerationResult(
                    text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="fake"
                ),
                GenerationResult(text="SELECT 1", finish_reason="stop", model_id="fake"),
                GenerationResult(text="SELECT * FROM nope", finish_reason="stop", model_id="fake"),
                GenerationResult(
                    text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="fake"
                ),
            ]
        )
        client = FakeModelClient(lambda _p: next(responses))
        db_root = _db_root(tmp_path)
        results = sample_k_candidates(
            _sample(),
            model_client=client,
            prompt="q0",
            db_root=db_root,
            gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
            k=4,
            temperature=0.8,
            max_tokens=256,
            generate_timeout_s=10.0,
        )
        assert len(results) == 4
        assert len(client.prompts) == 4
        assert results[0].comparison_result is ComparisonResult.EQUAL
        assert results[2].exec_code is not ErrorCode.EXEC_OK  # querying a nonexistent table

    def test_non_positive_k_rejected(self, tmp_path: Path) -> None:
        client = FakeModelClient(
            lambda _p: GenerationResult(text="x", finish_reason="stop", model_id="m")
        )
        with pytest.raises(ValueError, match="k must be positive"):
            sample_k_candidates(
                _sample(),
                model_client=client,
                prompt="q0",
                db_root=_db_root(tmp_path),
                gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
                k=0,
                temperature=0.8,
                max_tokens=256,
                generate_timeout_s=10.0,
            )


class TestCheckTrlAvailable:
    def test_reports_a_real_check_status(self) -> None:
        assert check_trl_available() in {CheckStatus.PASS, CheckStatus.SKIP}


class TestBuildDpoConfigKwargs:
    def test_renders_real_trl_dpoconfig_fields(self) -> None:
        config = DpoTrainingConfig(
            base_sft_adapter_path="artifacts/train/sft-qwen3.5-9b",
            lora_target_modules=["gate_proj", "q_proj"],
            beta=0.1,
            lr=5e-6,
            batch_size=2,
            seed=42,
            max_hours=4.0,
            output_dir="artifacts/train/dpo-qwen3.5-9b",
        )
        kwargs = build_dpo_config_kwargs(config)
        assert kwargs["beta"] == 0.1
        assert kwargs["learning_rate"] == 5e-6
        assert kwargs["seed"] == 42


class TestRunDpoTraining:
    def test_empty_preference_pairs_rejected(self) -> None:
        config = DpoTrainingConfig(
            base_sft_adapter_path="x",
            lora_target_modules=["gate_proj"],
            beta=0.1,
            lr=5e-6,
            batch_size=2,
            seed=42,
            max_hours=4.0,
            output_dir="out",
        )
        with pytest.raises(ValueError, match="empty"):
            run_dpo_training([], config)

    def test_raises_when_trl_not_available(self) -> None:
        if check_trl_available() is CheckStatus.PASS:
            pytest.skip("trl is actually installed in this environment")
        pairs = build_preference_dataset(
            {
                "q0": [
                    _prediction(predicted_sql="c0"),
                    _prediction(predicted_sql="w0", comparison_result=ComparisonResult.NOT_EQUAL),
                ]
            }
        ).all_pairs
        config = DpoTrainingConfig(
            base_sft_adapter_path="x",
            lora_target_modules=["gate_proj"],
            beta=0.1,
            lr=5e-6,
            batch_size=2,
            seed=42,
            max_hours=4.0,
            output_dir="out",
        )
        with pytest.raises(RuntimeError, match="cannot start DPO training"):
            run_dpo_training(pairs, config)
