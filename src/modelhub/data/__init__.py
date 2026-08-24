"""Dataset ingestion, normalization, versioning, and split integrity checks.

Public API: `NormalizedSample`/`Source`/`Split`/`Dialect` (the canonical
schema), `dedup_samples`, `assert_splits_disjoint`, `validate_gold_sql`,
`select_quick_eval_ids`, `export_safety_adversarial_set`,
`build_dataset_version` (the full pipeline).
"""

from modelhub.data.dedup import DedupResult, DedupStats, dedup_samples
from modelhub.data.gold_validation import GoldValidationSummary, validate_gold_sql
from modelhub.data.hashing import dataset_content_hash, question_hash, sql_hash
from modelhub.data.normalize import (
    normalize_bird_record,
    normalize_minidev_record,
    normalize_spider_record,
)
from modelhub.data.pipeline import DataBuildReport, build_dataset_version
from modelhub.data.quick_eval import select_quick_eval_ids
from modelhub.data.safety_adversarial import export_safety_adversarial_set
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.data.splits import assert_splits_disjoint

__all__ = [
    "DataBuildReport",
    "DedupResult",
    "DedupStats",
    "Dialect",
    "GoldValidationSummary",
    "NormalizedSample",
    "Source",
    "Split",
    "assert_splits_disjoint",
    "build_dataset_version",
    "dataset_content_hash",
    "dedup_samples",
    "export_safety_adversarial_set",
    "normalize_bird_record",
    "normalize_minidev_record",
    "normalize_spider_record",
    "question_hash",
    "select_quick_eval_ids",
    "sql_hash",
    "validate_gold_sql",
]
