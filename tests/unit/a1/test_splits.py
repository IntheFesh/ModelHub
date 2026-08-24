import pytest

from modelhub.common.errors import ErrorCode, ModelHubError
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.data.splits import assert_splits_disjoint, split_of


def _sample(sample_id: str, question: str, db_id: str, split: Split) -> NormalizedSample:
    return NormalizedSample(
        sample_id=sample_id,
        db_id=db_id,
        question=question,
        evidence=None,
        gold_sql="SELECT 1",
        difficulty=None,
        source=Source.BIRD,
        split=split,
        dialect=Dialect.SQLITE,
    )


def test_disjoint_splits_pass() -> None:
    train = [_sample("t1", "Q1", "db1", Split.TRAIN)]
    dev = [_sample("d1", "Q2", "db1", Split.DEV)]
    test = [_sample("x1", "Q3", "db1", Split.TEST)]
    assert_splits_disjoint(train, dev, test)  # must not raise


def test_same_question_different_db_is_not_a_leak() -> None:
    train = [_sample("t1", "How many rows?", "db1", Split.TRAIN)]
    dev = [_sample("d1", "How many rows?", "db2", Split.DEV)]
    assert_splits_disjoint(train, dev, [])  # different db_id -> not a leak


def test_same_question_same_db_across_splits_is_a_leak() -> None:
    train = [_sample("t1", "How many students?", "school", Split.TRAIN)]
    dev = [_sample("d1", "How many students?", "school", Split.DEV)]
    with pytest.raises(ModelHubError) as exc_info:
        assert_splits_disjoint(train, dev, [])
    assert exc_info.value.code is ErrorCode.DATA_SPLIT_LEAK
    assert "train∩dev" in exc_info.value.context["leaks"]


def test_split_of_filters_by_split() -> None:
    samples = [
        _sample("t1", "Q1", "db1", Split.TRAIN),
        _sample("d1", "Q2", "db1", Split.DEV),
    ]
    assert [s.sample_id for s in split_of(samples, Split.TRAIN)] == ["t1"]
    assert [s.sample_id for s in split_of(samples, Split.DEV)] == ["d1"]
