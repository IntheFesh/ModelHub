from pathlib import Path

from modelhub.data.gold_validation import validate_gold_sql, write_gold_exec_failures
from modelhub.data.schema import NormalizedSample


def test_gold_validation_separates_ok_and_failed(
    bird_train_samples: list[NormalizedSample], bird_db_root: Path
) -> None:
    # fixture: samples 1,2 execute fine against school.sqlite; sample 3
    # (enrollments_typo) does not — this mirrors BIRD's real 425 failing
    # gold-SQL rows, at fixture scale.
    summary, failures = validate_gold_sql(bird_train_samples, db_root=bird_db_root)
    assert summary.total == 3
    assert summary.exec_ok == 2
    assert summary.failed == 1
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "bird:train:3"
    assert failures[0]["code"] == "SEMANTIC"


def test_gold_validation_failures_are_never_silently_dropped(
    bird_train_samples: list[NormalizedSample], bird_db_root: Path, tmp_path: Path
) -> None:
    _summary, failures = validate_gold_sql(bird_train_samples, db_root=bird_db_root)
    out_path = write_gold_exec_failures(failures, dataset_version="test-v1")
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == len(failures) == 1
    assert "enrollments_typo" in lines[0]


def test_write_gold_exec_failures_writes_present_empty_file_when_none() -> None:
    # An empty file must exist (not "missing"), so "no failures" is never
    # confused with "the validation step never ran".
    path = write_gold_exec_failures([], dataset_version="test-v2-clean")
    assert path.exists()
    assert path.read_text() == ""
