"""Aggregate a list of PredictionRecords into report-ready metrics.

CLAUDE.md/A4: HARNESS_* proportion > 1% must refuse to produce a report
at all (not silently give a low score) — a flood of harness errors means
the numbers below are meaningless, not "bad". This module raises rather
than returning a degraded metrics object in that case.
"""

from __future__ import annotations

from collections import Counter

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.records import PredictionRecord

HARNESS_ERROR_FLOOD_THRESHOLD = 0.01


class HarnessErrorFloodError(Exception):
    """Raised instead of producing metrics when HARNESS_* exceeds 1% of samples."""

    def __init__(self, harness_count: int, total: int, breakdown: dict[str, int]) -> None:
        self.harness_count = harness_count
        self.total = total
        self.breakdown = breakdown
        rate = harness_count / total if total else 0.0
        super().__init__(
            f"HARNESS_* errors are {rate:.1%} of {total} samples (threshold "
            f"{HARNESS_ERROR_FLOOD_THRESHOLD:.0%}) — refusing to produce a report. "
            f"breakdown={breakdown}"
        )


class EvalMetrics(ModelHubBaseConfig):
    total_samples: int
    denominator: int  # excludes OUTPUT_TRUNCATED and every UNDECIDABLE case
    excluded_output_truncated: int
    excluded_undecidable: int
    exec_ok_count: int
    equal_count: int
    execution_accuracy: float | None  # None if denominator is 0 — never a fake 0.0
    syntax_valid_rate: float  # EXEC_OK / total_samples (a separate, always-computable metric)
    error_breakdown: dict[str, int]
    output_truncated_rate: float
    undecidable_rate: float
    avg_elapsed_s: float
    by_difficulty: dict[str, DifficultySlice]
    by_db: dict[str, DifficultySlice]


class DifficultySlice(ModelHubBaseConfig):
    total: int
    denominator: int
    equal_count: int
    execution_accuracy: float | None


def _slice_metrics(records: list[PredictionRecord]) -> DifficultySlice:
    denom_records = [r for r in records if r.counts_toward_denominator]
    equal_count = sum(1 for r in denom_records if r.comparison_result is ComparisonResult.EQUAL)
    denom = len(denom_records)
    return DifficultySlice(
        total=len(records),
        denominator=denom,
        equal_count=equal_count,
        execution_accuracy=(equal_count / denom) if denom > 0 else None,
    )


def compute_metrics(records: list[PredictionRecord]) -> EvalMetrics:
    total = len(records)
    if total == 0:
        raise ValueError("cannot compute metrics over an empty prediction set")

    harness_count = sum(1 for r in records if r.is_harness_error)
    if harness_count / total > HARNESS_ERROR_FLOOD_THRESHOLD:
        breakdown = dict(Counter(r.exec_code.value for r in records if r.is_harness_error))
        raise HarnessErrorFloodError(harness_count, total, breakdown)

    denom_records = [r for r in records if r.counts_toward_denominator]
    equal_count = sum(1 for r in denom_records if r.comparison_result is ComparisonResult.EQUAL)
    denom = len(denom_records)

    output_truncated_count = sum(1 for r in records if r.is_output_truncated)
    undecidable_count = sum(
        1 for r in records if r.comparison_result is ComparisonResult.UNDECIDABLE
    )
    exec_ok_count = sum(1 for r in records if r.exec_code is ErrorCode.EXEC_OK)

    by_difficulty: dict[str, DifficultySlice] = {}
    for difficulty in {r.difficulty or "unlabeled" for r in records}:
        by_difficulty[difficulty] = _slice_metrics(
            [r for r in records if (r.difficulty or "unlabeled") == difficulty]
        )

    by_db: dict[str, DifficultySlice] = {}
    for db_id in {r.db_id for r in records}:
        by_db[db_id] = _slice_metrics([r for r in records if r.db_id == db_id])

    return EvalMetrics(
        total_samples=total,
        denominator=denom,
        excluded_output_truncated=output_truncated_count,
        excluded_undecidable=undecidable_count,
        exec_ok_count=exec_ok_count,
        equal_count=equal_count,
        execution_accuracy=(equal_count / denom) if denom > 0 else None,
        syntax_valid_rate=exec_ok_count / total,
        error_breakdown=dict(Counter(r.exec_code.value for r in records)),
        output_truncated_rate=output_truncated_count / total,
        undecidable_rate=undecidable_count / total,
        avg_elapsed_s=sum(r.elapsed_s for r in records) / total,
        by_difficulty=by_difficulty,
        by_db=by_db,
    )
