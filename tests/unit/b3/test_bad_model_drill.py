"""Unit tests for scripts/bad_model_drill.py's reusable helper
functions — the real subprocess-free pieces, not `make bad-model-drill`
itself (hand-run and verified for real — see docs/build-log.md)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import bad_model_drill
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.data.schema import Source
from modelhub.gate.types import GateDecision


class TestBuildDrillDb:
    def test_creates_a_real_queryable_sqlite_db(self, tmp_path: Path) -> None:
        db_path = bad_model_drill._build_drill_db(tmp_path / "dbs")
        assert db_path.is_file()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id, name FROM students ORDER BY id").fetchall()
        conn.close()
        assert rows == [(1, "Ada"), (2, "Grace")]


class TestPredictionHelpers:
    def test_correct_prediction_is_equal(self) -> None:
        record = bad_model_drill._prediction("s0", correct=True)
        assert record.comparison_result is ComparisonResult.EQUAL
        assert record.exec_code is ErrorCode.EXEC_OK

    def test_incorrect_prediction_is_not_equal(self) -> None:
        record = bad_model_drill._prediction("s0", correct=False)
        assert record.comparison_result is ComparisonResult.NOT_EQUAL

    def test_unsafe_prediction_has_no_comparison_result(self) -> None:
        record = bad_model_drill._unsafe_prediction("s0")
        assert record.exec_code is ErrorCode.UNSAFE_STATEMENT
        assert record.comparison_result is None
        assert record.counts_toward_denominator  # a real wrong attempt, not excluded


class TestCrudSample:
    def test_source_is_minidev_crud(self) -> None:
        sample = bad_model_drill._crud_sample("c0", "DELETE FROM students")
        assert sample.source is Source.MINIDEV_CRUD
        assert sample.gold_sql == "DELETE FROM students"


class TestDrillCkptA:
    """`_drill_ckpt_a` end to end against a real tmp SQLite db."""

    def test_genuinely_low_accuracy_is_rejected(self, tmp_path: Path) -> None:
        bad_model_drill._build_drill_db(tmp_path)
        config = bad_model_drill._load_admission_config()
        verdict = bad_model_drill._drill_ckpt_a(tmp_path, config)
        assert verdict.decision is GateDecision.REJECT
        result = verdict.result_for("accuracy")
        assert result is not None
        assert result.decision is GateDecision.REJECT


class TestDrillCkptB:
    def test_regression_is_named_and_rejected(self, tmp_path: Path) -> None:
        bad_model_drill._build_drill_db(tmp_path)
        config = bad_model_drill._load_admission_config()
        verdict = bad_model_drill._drill_ckpt_b(tmp_path, config)
        assert verdict.decision is GateDecision.REJECT
        result = verdict.result_for("regression")
        assert result is not None
        assert result.decision is GateDecision.REJECT
        assert "complex0" in result.metrics["regressed_sample_ids"]


class TestDrillCkptD:
    def test_safety_gate_passes_but_accuracy_gate_rejects(self, tmp_path: Path) -> None:
        bad_model_drill._build_drill_db(tmp_path)
        config = bad_model_drill._load_admission_config()
        verdict = bad_model_drill._drill_ckpt_d(tmp_path, config)
        safety_result = verdict.result_for("safety")
        accuracy_result = verdict.result_for("accuracy")
        assert safety_result is not None and safety_result.decision is GateDecision.PASS
        assert accuracy_result is not None and accuracy_result.decision is GateDecision.REJECT
        assert verdict.decision is GateDecision.REJECT


class TestRenderReadme:
    def test_readme_mentions_every_profile_and_verdict(self, tmp_path: Path) -> None:
        bad_model_drill._build_drill_db(tmp_path)
        config = bad_model_drill._load_admission_config()
        verdicts = {
            bad_model_drill.BadModelProfile.CKPT_A_UNDERFIT: bad_model_drill._drill_ckpt_a(
                tmp_path, config
            ),
            bad_model_drill.BadModelProfile.CKPT_B_REGRESSION: bad_model_drill._drill_ckpt_b(
                tmp_path, config
            ),
            bad_model_drill.BadModelProfile.CKPT_D_SAFETY: bad_model_drill._drill_ckpt_d(
                tmp_path, config
            ),
        }
        readme = bad_model_drill._render_readme(verdicts)
        for profile in bad_model_drill.BadModelProfile:
            assert profile.value in readme
        assert readme.count("REJECT") == 3
