"""Unit tests for bench/experiments/constrained_decoding.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import fixed_response_client
from tests.unit.a11.fakes import build_school_db, select_sample

from modelhub.bench.experiments.constrained_decoding import (
    ConstrainedDecodingResult,
    compile_sqlite_select_grammar,
    syntax_legality_rate,
)
from modelhub.common.capability import CheckStatus
from modelhub.eval.metrics import EvalMetrics, compute_metrics
from modelhub.eval.runner import run_eval


class TestCompileSqliteSelectGrammar:
    def test_skips_when_xgrammar_not_installed(self) -> None:
        result = compile_sqlite_select_grammar()
        assert result.status == CheckStatus.SKIP
        assert "xgrammar" in result.detail


class TestSyntaxLegalityRate:
    def test_empty_predictions_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty prediction set"):
            syntax_legality_rate([])

    def test_all_syntax_valid_is_one(self, tmp_path: Path) -> None:
        db_root = build_school_db(tmp_path)
        samples = [select_sample("s0")]
        client = fixed_response_client("SELECT id FROM students")
        predictions = run_eval(
            samples,
            model_client=client,
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "predictions.jsonl",
        )
        assert syntax_legality_rate(predictions) == 1.0

    def test_a_syntax_error_lowers_the_rate(self, tmp_path: Path) -> None:
        db_root = build_school_db(tmp_path)
        samples = [select_sample("s0"), select_sample("s1")]
        client = fixed_response_client("SELECT FROM WHERE")  # deliberately invalid SQL
        predictions = run_eval(
            samples,
            model_client=client,
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "predictions.jsonl",
        )
        assert syntax_legality_rate(predictions) == 0.0


class TestConstrainedDecodingResult:
    def test_syntax_legality_improvement(self, tmp_path: Path) -> None:
        db_root = build_school_db(tmp_path)
        samples = [select_sample("s0")]

        unconstrained_predictions = run_eval(
            samples,
            model_client=fixed_response_client("SELECT FROM WHERE"),
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "unconstrained.jsonl",
        )
        constrained_predictions = run_eval(
            samples,
            model_client=fixed_response_client("SELECT id FROM students"),
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "constrained.jsonl",
        )
        result = ConstrainedDecodingResult(
            unconstrained_syntax_legality_rate=syntax_legality_rate(unconstrained_predictions),
            constrained_syntax_legality_rate=syntax_legality_rate(constrained_predictions),
            unconstrained_metrics=compute_metrics(unconstrained_predictions),
            constrained_metrics=compute_metrics(constrained_predictions),
            unconstrained_output_tokens_per_s=100.0,
            constrained_output_tokens_per_s=90.0,
        )
        assert result.syntax_legality_rate_improvement == pytest.approx(1.0)
        assert result.throughput_cost_pct == pytest.approx(10.0)
        assert result.execution_accuracy_change == pytest.approx(1.0)

    def test_throughput_cost_none_when_baseline_missing(self) -> None:
        empty_metrics = EvalMetrics(
            total_samples=0,
            denominator=0,
            excluded_output_truncated=0,
            excluded_undecidable=0,
            exec_ok_count=0,
            equal_count=0,
            execution_accuracy=None,
            syntax_valid_rate=0.0,
            error_breakdown={},
            output_truncated_rate=0.0,
            undecidable_rate=0.0,
            avg_elapsed_s=0.0,
            by_difficulty={},
            by_db={},
        )
        result = ConstrainedDecodingResult(
            unconstrained_syntax_legality_rate=0.0,
            constrained_syntax_legality_rate=0.0,
            unconstrained_metrics=empty_metrics,
            constrained_metrics=empty_metrics,
            unconstrained_output_tokens_per_s=None,
            constrained_output_tokens_per_s=None,
        )
        assert result.throughput_cost_pct is None
        assert result.execution_accuracy_change is None
