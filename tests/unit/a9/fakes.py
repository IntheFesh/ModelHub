"""Shared builders for gate/registry tests — CLAUDE.md §1.4: these are
plain construction helpers around real project types, not mocks/fakes
standing in for logic under test."""

from __future__ import annotations

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.records import PredictionRecord


def prediction(sample_id: str, **overrides: object) -> PredictionRecord:
    defaults: dict[str, object] = {
        "sample_id": sample_id,
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


def crud_sample(sample_id: str, gold_sql: str, db_id: str = "school") -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": "irrelevant for this test",
            "evidence": None,
            "gold_sql": gold_sql,
            "difficulty": None,
            "source": Source.MINIDEV_CRUD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )
