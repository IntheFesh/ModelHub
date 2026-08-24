"""Evaluation runner and report generation (quick-tier / full-tier).

Public API: `run_eval`/`run_one_sample`/`load_predictions` (runner.py),
`compute_metrics`/`EvalMetrics`/`HarnessErrorFloodError` (metrics.py),
`write_report`/`render_report_markdown`/`assert_report_ready` (report.py),
`GoldExecCache` (gold_cache.py), `ModelClient`/`HttpModelClient`/
`GenerationResult` (model_client.py), `PredictionRecord` (records.py).
"""

from modelhub.eval.gold_cache import GoldExecCache, cache_key, db_file_hash
from modelhub.eval.metrics import (
    HARNESS_ERROR_FLOOD_THRESHOLD,
    DifficultySlice,
    EvalMetrics,
    HarnessErrorFloodError,
    compute_metrics,
)
from modelhub.eval.model_client import GenerationResult, HttpModelClient, ModelClient
from modelhub.eval.records import PredictionRecord
from modelhub.eval.report import (
    REQUIRED_MANIFEST_FIELDS,
    assert_report_ready,
    render_report_markdown,
    report_path,
    write_report,
)
from modelhub.eval.runner import EvalTier, load_predictions, run_eval, run_one_sample

__all__ = [
    "HARNESS_ERROR_FLOOD_THRESHOLD",
    "REQUIRED_MANIFEST_FIELDS",
    "DifficultySlice",
    "EvalMetrics",
    "EvalTier",
    "GenerationResult",
    "GoldExecCache",
    "HarnessErrorFloodError",
    "HttpModelClient",
    "ModelClient",
    "PredictionRecord",
    "assert_report_ready",
    "cache_key",
    "compute_metrics",
    "db_file_hash",
    "load_predictions",
    "render_report_markdown",
    "report_path",
    "run_eval",
    "run_one_sample",
    "write_report",
]
