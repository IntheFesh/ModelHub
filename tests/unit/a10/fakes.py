"""Shared builders for A10 (canary rollout) tests — CLAUDE.md §1.4: plain
construction helpers around real project types, not mocks standing in for
logic under test. Reuses A9's `prediction()` builder pattern rather than
duplicating it, since `PredictionRecord` is exactly what `online_sampling.py`
consumes.
"""

from __future__ import annotations

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.records import PredictionRecord


def healthy_prediction(sample_id: str, *, elapsed_s: float = 0.1) -> PredictionRecord:
    return PredictionRecord.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "difficulty": "simple",
            "predicted_sql": "SELECT 1",
            "finish_reason": "stop",
            "exec_code": ErrorCode.EXEC_OK,
            "comparison_result": ComparisonResult.EQUAL,
            "elapsed_s": elapsed_s,
        }
    )


def failing_prediction(sample_id: str, *, elapsed_s: float = 0.1) -> PredictionRecord:
    """A model-side SQL error (SYNTAX) — a real degraded-canary signal,
    not a harness-side failure."""
    return PredictionRecord.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "difficulty": "simple",
            "predicted_sql": "SELECT FROM WHERE",
            "finish_reason": "stop",
            "exec_code": ErrorCode.SYNTAX,
            "comparison_result": None,
            "elapsed_s": elapsed_s,
        }
    )


def harness_error_prediction(sample_id: str, *, elapsed_s: float = 0.1) -> PredictionRecord:
    return PredictionRecord.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "difficulty": "simple",
            "predicted_sql": "SELECT 1",
            "finish_reason": "stop",
            "exec_code": ErrorCode.HARNESS_DB_UNAVAILABLE,
            "comparison_result": None,
            "elapsed_s": elapsed_s,
        }
    )
