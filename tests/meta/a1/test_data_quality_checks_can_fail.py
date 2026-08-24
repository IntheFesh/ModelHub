"""A1 acceptance bullet: "断言三划分交集为空" + gold-exec failures counted,
never silently dropped. Proves both checks can actually go red.
"""

from pathlib import Path

import pytest

from modelhub.common.errors import ErrorCode, ModelHubError
from modelhub.data.gold_validation import validate_gold_sql, write_gold_exec_failures
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.data.splits import assert_splits_disjoint


def _sample(
    sample_id: str, question: str, db_id: str, split: Split, sql: str = "SELECT 1"
) -> NormalizedSample:
    return NormalizedSample(
        sample_id=sample_id,
        db_id=db_id,
        question=question,
        evidence=None,
        gold_sql=sql,
        difficulty=None,
        source=Source.BIRD,
        split=split,
        dialect=Dialect.SQLITE,
    )


def test_split_leak_detector_fires_on_injected_leak() -> None:
    leaked_train = [_sample("t1", "duplicate question", "school", Split.TRAIN)]
    leaked_dev = [_sample("d1", "duplicate question", "school", Split.DEV)]
    with pytest.raises(ModelHubError) as exc_info:
        assert_splits_disjoint(leaked_train, leaked_dev, [])
    assert exc_info.value.code is ErrorCode.DATA_SPLIT_LEAK


def test_split_leak_detector_stays_quiet_on_genuinely_disjoint_splits() -> None:
    # Negative case — proves the detector isn't just always-raise.
    train = [_sample("t1", "question A", "school", Split.TRAIN)]
    dev = [_sample("d1", "question B", "school", Split.DEV)]
    assert_splits_disjoint(train, dev, [])


def test_gold_exec_failures_all_fail_is_counted_not_hidden(bird_db_root: Path) -> None:
    # Inject samples whose gold SQL is guaranteed to fail (unknown table),
    # like a subset of BIRD's real 425 non-executing gold rows.
    doomed = [
        _sample(f"doomed{i}", f"q{i}", "school", Split.TRAIN, sql=f"SELECT * FROM ghost_table_{i}")
        for i in range(5)
    ]
    summary, failures = validate_gold_sql(doomed, db_root=bird_db_root)
    assert summary.failed == 5
    assert summary.exec_ok == 0
    assert len(failures) == 5  # every failure individually present, not just a count

    out_path = write_gold_exec_failures(failures, dataset_version="meta-test-doomed")
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5
