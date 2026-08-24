"""Unit tests for eval/records.py: PredictionRecord's denominator logic.

This is the single most consequential piece of logic in A4 — get
`counts_toward_denominator` wrong and every accuracy number downstream
(reports, gate decisions, GRPO reward masking) is silently wrong in a way
that looks like a real number.
"""

from __future__ import annotations

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.eval.records import PredictionRecord


def _record(**overrides: object) -> PredictionRecord:
    defaults: dict[str, object] = {
        "sample_id": "s1",
        "db_id": "school",
        "difficulty": "simple",
        "predicted_sql": "SELECT 1",
        "finish_reason": "stop",
        "exec_code": ErrorCode.EXEC_OK,
        "comparison_result": ComparisonResult.EQUAL,
        "elapsed_s": 0.1,
    }
    defaults.update(overrides)
    return PredictionRecord.model_validate(defaults)


def test_output_truncated_flag() -> None:
    r = _record(
        exec_code=ErrorCode.OUTPUT_TRUNCATED, comparison_result=None, finish_reason="length"
    )
    assert r.is_output_truncated
    assert not r.is_harness_error


def test_harness_error_flag_covers_both_codes() -> None:
    for code in (ErrorCode.HARNESS_DB_UNAVAILABLE, ErrorCode.HARNESS_INTERNAL):
        r = _record(exec_code=code, comparison_result=None)
        assert r.is_harness_error
        assert not r.is_output_truncated


def test_model_error_codes_are_neither_truncated_nor_harness() -> None:
    for code in (ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT):
        r = _record(exec_code=code, comparison_result=None)
        assert not r.is_output_truncated
        assert not r.is_harness_error


def test_output_truncated_excluded_from_denominator() -> None:
    r = _record(
        exec_code=ErrorCode.OUTPUT_TRUNCATED, comparison_result=None, finish_reason="length"
    )
    assert not r.counts_toward_denominator


def test_harness_error_excluded_from_denominator_even_with_comparison_result() -> None:
    # Should never happen in practice (a harness error means execution never
    # reached comparison), but the property must not trust comparison_result
    # blindly — harness-ness overrides it either way.
    r = _record(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=ComparisonResult.EQUAL)
    assert not r.counts_toward_denominator


def test_undecidable_excluded_from_denominator() -> None:
    r = _record(
        exec_code=ErrorCode.EXEC_OK,
        comparison_result=ComparisonResult.UNDECIDABLE,
        undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
    )
    assert not r.counts_toward_denominator


def test_model_syntax_error_counts_as_zero_credit_attempt() -> None:
    # CLAUDE.md §2.3: the model's own mistake, not a harness failure —
    # counts toward the denominator (as a miss), unlike OUTPUT_TRUNCATED
    # or UNDECIDABLE.
    r = _record(exec_code=ErrorCode.SYNTAX, comparison_result=None)
    assert r.counts_toward_denominator


def test_equal_and_not_equal_both_count() -> None:
    for result in (ComparisonResult.EQUAL, ComparisonResult.NOT_EQUAL):
        r = _record(exec_code=ErrorCode.EXEC_OK, comparison_result=result)
        assert r.counts_toward_denominator
