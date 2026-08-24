"""Unit tests for eval/metrics.py."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.eval.metrics import HarnessErrorFloodError, compute_metrics
from modelhub.eval.records import PredictionRecord


def _record(sample_id: str, **overrides: object) -> PredictionRecord:
    defaults: dict[str, object] = {
        "sample_id": sample_id,
        "db_id": "db1",
        "difficulty": "simple",
        "predicted_sql": "SELECT 1",
        "finish_reason": "stop",
        "exec_code": ErrorCode.EXEC_OK,
        "comparison_result": ComparisonResult.EQUAL,
        "elapsed_s": 0.1,
    }
    defaults.update(overrides)
    return PredictionRecord.model_validate(defaults)


def test_compute_metrics_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])


def test_compute_metrics_basic_accuracy() -> None:
    records = [
        _record("s1", comparison_result=ComparisonResult.EQUAL),
        _record("s2", comparison_result=ComparisonResult.NOT_EQUAL),
        _record("s3", exec_code=ErrorCode.SYNTAX, comparison_result=None),
    ]
    m = compute_metrics(records)
    assert m.total_samples == 3
    assert m.denominator == 3
    assert m.equal_count == 1
    assert m.execution_accuracy == pytest.approx(1 / 3)
    assert m.exec_ok_count == 2
    assert m.syntax_valid_rate == pytest.approx(2 / 3)


def test_output_truncated_excluded_from_denominator_and_accuracy() -> None:
    records = [
        _record("s1", comparison_result=ComparisonResult.EQUAL),
        _record(
            "s2",
            exec_code=ErrorCode.OUTPUT_TRUNCATED,
            comparison_result=None,
            finish_reason="length",
        ),
    ]
    m = compute_metrics(records)
    assert m.total_samples == 2
    assert m.denominator == 1
    assert m.excluded_output_truncated == 1
    assert m.execution_accuracy == 1.0
    assert m.output_truncated_rate == pytest.approx(0.5)


def test_undecidable_excluded_from_denominator() -> None:
    records = [
        _record("s1", comparison_result=ComparisonResult.EQUAL),
        _record(
            "s2",
            comparison_result=ComparisonResult.UNDECIDABLE,
            undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
        ),
    ]
    m = compute_metrics(records)
    assert m.denominator == 1
    assert m.excluded_undecidable == 1
    assert m.undecidable_rate == pytest.approx(0.5)
    assert m.execution_accuracy == 1.0


def test_execution_accuracy_is_none_when_denominator_is_zero() -> None:
    records = [
        _record(
            "s1",
            exec_code=ErrorCode.OUTPUT_TRUNCATED,
            comparison_result=None,
            finish_reason="length",
        )
    ]
    m = compute_metrics(records)
    assert m.denominator == 0
    assert m.execution_accuracy is None


def test_harness_error_flood_raises_above_one_percent() -> None:
    records = [_record(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(98)]
    records += [
        _record(f"h{i}", exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        for i in range(2)
    ]
    with pytest.raises(HarnessErrorFloodError) as exc_info:
        compute_metrics(records)
    assert exc_info.value.harness_count == 2
    assert exc_info.value.total == 100


def test_harness_error_exactly_at_threshold_does_not_raise() -> None:
    records = [_record(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(99)]
    records += [_record("h0", exec_code=ErrorCode.HARNESS_DB_UNAVAILABLE, comparison_result=None)]
    m = compute_metrics(records)  # 1/100 == 1%, threshold is "> 1%", must not raise
    assert m.total_samples == 100


def test_by_difficulty_and_by_db_slices() -> None:
    records = [
        _record("s1", db_id="db1", difficulty="simple", comparison_result=ComparisonResult.EQUAL),
        _record(
            "s2", db_id="db1", difficulty="simple", comparison_result=ComparisonResult.NOT_EQUAL
        ),
        _record("s3", db_id="db2", difficulty="hard", comparison_result=ComparisonResult.EQUAL),
    ]
    m = compute_metrics(records)
    assert m.by_difficulty["simple"].total == 2
    assert m.by_difficulty["simple"].equal_count == 1
    assert m.by_difficulty["simple"].execution_accuracy == pytest.approx(0.5)
    assert m.by_difficulty["hard"].execution_accuracy == 1.0
    assert m.by_db["db1"].total == 2
    assert m.by_db["db2"].total == 1


def test_unlabeled_difficulty_bucketed_separately() -> None:
    records = [
        _record("s1", difficulty=None, comparison_result=ComparisonResult.EQUAL),
        _record("s2", difficulty="simple", comparison_result=ComparisonResult.EQUAL),
    ]
    m = compute_metrics(records)
    assert "unlabeled" in m.by_difficulty
    assert m.by_difficulty["unlabeled"].total == 1
