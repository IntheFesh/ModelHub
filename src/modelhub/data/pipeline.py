"""Orchestrates one dataset-version build: dedup -> split-check -> hash ->
[optional gold validation] -> data-quality report.

This is itself a number-producing execution (dedup counts, split sizes,
difficulty distribution, gold-failure counts) so CLAUDE.md §3.1 applies:
it writes both a `RunManifest` (so downstream code can pollution-check it
like any other run) and a data-specific `DataBuildReport` with the counts
a run manifest's fixed schema has no field for.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from modelhub.common.atomic_io import atomic_write_json
from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.dedup import DedupStats, dedup_samples
from modelhub.data.gold_validation import (
    GoldValidationSummary,
    validate_gold_sql,
    write_gold_exec_failures,
)
from modelhub.data.hashing import dataset_content_hash
from modelhub.data.schema import NormalizedSample, Split
from modelhub.data.splits import assert_splits_disjoint, split_of


class DataBuildReport(ModelHubBaseConfig):
    dataset_version: str
    dataset_hash: str
    total_samples: int
    per_split_counts: dict[str, int]
    per_difficulty_counts: dict[str, int]
    dedup_stats: DedupStats
    gold_validation: GoldValidationSummary | None


def build_dataset_version(
    raw_samples: Sequence[NormalizedSample],
    *,
    dataset_version: str,
    db_root: Path | None = None,
    validate_gold: bool = True,
    artifacts_root: Path = Path("artifacts/data"),
) -> DataBuildReport:
    """Run the full A1 pipeline over an already-normalized sample list.

    `db_root`: directory of `<db_id>/<db_id>.sqlite` files for gold-SQL
    execution validation. If `validate_gold` is True but `db_root` is None
    or doesn't exist, this raises rather than silently skipping — CLAUDE.md
    §1.3: a check that can't run must never be treated as having passed.
    """
    dedup_result = dedup_samples(list(raw_samples))
    samples = dedup_result.samples

    train = split_of(samples, Split.TRAIN)
    dev = split_of(samples, Split.DEV)
    test = split_of(samples, Split.TEST)
    assert_splits_disjoint(train, dev, test)

    per_split_counts = {"train": len(train), "dev": len(dev), "test": len(test)}
    per_difficulty_counts: dict[str, int] = {}
    for s in samples:
        key = s.difficulty or "unlabeled"
        per_difficulty_counts[key] = per_difficulty_counts.get(key, 0) + 1

    gold_summary: GoldValidationSummary | None = None
    if validate_gold:
        if db_root is None or not db_root.is_dir():
            raise FileNotFoundError(
                f"validate_gold=True but db_root={db_root!r} is not a directory. "
                f"Pass validate_gold=False explicitly if gold-SQL execution "
                f"validation genuinely cannot run yet (e.g. no networked machine "
                f"to download the .sqlite files) — do not let this silently skip."
            )
        gold_summary, failures = validate_gold_sql(samples, db_root=db_root)
        if failures:
            write_gold_exec_failures(
                failures, dataset_version=dataset_version, artifacts_root=artifacts_root
            )

    content_hash = dataset_content_hash(samples)

    report = DataBuildReport(
        dataset_version=dataset_version,
        dataset_hash=content_hash,
        total_samples=len(samples),
        per_split_counts=per_split_counts,
        per_difficulty_counts=per_difficulty_counts,
        dedup_stats=dedup_result.stats,
        gold_validation=gold_summary,
    )
    return report


def write_data_build_report(
    report: DataBuildReport, *, artifacts_root: Path = Path("artifacts/data")
) -> Path:
    path = artifacts_root / report.dataset_version / "dataset_summary.json"
    atomic_write_json(path, report.model_dump(mode="json"))
    return path
