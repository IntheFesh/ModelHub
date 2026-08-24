from pathlib import Path

import pytest

from modelhub.data.pipeline import build_dataset_version, write_data_build_report
from modelhub.data.schema import NormalizedSample


def test_build_dataset_version_end_to_end(
    bird_train_samples: list[NormalizedSample],
    bird_dev_samples: list[NormalizedSample],
    bird_db_root: Path,
    tmp_path: Path,
) -> None:
    report = build_dataset_version(
        bird_train_samples + bird_dev_samples,
        dataset_version="test-build-v1",
        db_root=bird_db_root,
        artifacts_root=tmp_path / "artifacts" / "data",
    )
    # 1 question-level dup between train q1 and dev q1 -> 5 - 1 = 4 kept
    assert report.total_samples == 4
    assert report.per_split_counts["train"] == 3
    assert report.per_split_counts["dev"] == 1  # the dup (dev's copy) was dropped
    assert report.gold_validation is not None
    assert report.gold_validation.failed == 1  # the enrollments_typo sample
    assert report.dataset_hash.startswith("sha256:")

    path = write_data_build_report(report, artifacts_root=tmp_path / "artifacts" / "data")
    assert path.exists()


def test_build_dataset_version_requires_explicit_opt_out_of_gold_validation(
    bird_train_samples: list[NormalizedSample],
) -> None:
    # CLAUDE.md §1.3: a check that can't run must never silently pass.
    with pytest.raises(FileNotFoundError):
        build_dataset_version(bird_train_samples, dataset_version="test-build-v2", db_root=None)

    # Explicitly opting out is allowed and does not raise.
    report = build_dataset_version(
        bird_train_samples, dataset_version="test-build-v3", db_root=None, validate_gold=False
    )
    assert report.gold_validation is None
