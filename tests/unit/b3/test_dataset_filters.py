"""Unit tests for train/bad_models/dataset_filters.py."""

from __future__ import annotations

import pytest

from modelhub.data.schema import NormalizedSample, Source, Split
from modelhub.train.bad_models.dataset_filters import filter_easy_only, inject_crud_samples


def _sample(sample_id: str, *, source: Source, difficulty: str | None) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "question": "q",
            "evidence": None,
            "gold_sql": "SELECT 1",
            "difficulty": difficulty,
            "source": source,
            "split": Split.TRAIN,
        }
    )


def _crud_sample(sample_id: str, gold_sql: str) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "question": "q",
            "evidence": None,
            "gold_sql": gold_sql,
            "difficulty": None,
            "source": Source.MINIDEV_CRUD,
            "split": Split.DEV,
        }
    )


class TestFilterEasyOnly:
    def test_keeps_spider_easy_and_bird_simple(self) -> None:
        samples = [
            _sample("s1", source=Source.SPIDER, difficulty="easy"),
            _sample("s2", source=Source.BIRD, difficulty="simple"),
            _sample("s3", source=Source.SPIDER, difficulty="hard"),
            _sample("s4", source=Source.BIRD, difficulty="challenging"),
        ]
        filtered = filter_easy_only(samples)
        assert {s.sample_id for s in filtered} == {"s1", "s2"}

    def test_bird_easy_is_not_matched_by_spider_threshold(self) -> None:
        # "easy" only counts for Spider — BIRD's own vocab is "simple".
        samples = [_sample("s1", source=Source.BIRD, difficulty="easy")]
        with pytest.raises(ValueError, match="matched 0"):
            filter_easy_only(samples)

    def test_empty_result_raises(self) -> None:
        samples = [_sample("s1", source=Source.SPIDER, difficulty="hard")]
        with pytest.raises(ValueError, match="matched 0"):
            filter_easy_only(samples)


class TestInjectCrudSamples:
    def test_injects_requested_count(self) -> None:
        base = [_sample("s1", source=Source.BIRD, difficulty="simple")]
        crud = [_crud_sample(f"crud{i}", "DELETE FROM students") for i in range(5)]
        result = inject_crud_samples(base, crud, count=3)
        assert len(result) == 4
        assert sum(1 for s in result if s.source is Source.MINIDEV_CRUD) == 3

    def test_non_positive_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            inject_crud_samples([], [], count=0)

    def test_insufficient_crud_samples_rejected(self) -> None:
        crud = [_crud_sample("crud0", "DELETE FROM students")]
        with pytest.raises(ValueError, match="only 1"):
            inject_crud_samples([], crud, count=5)

    def test_wrong_source_crud_sample_rejected(self) -> None:
        wrong = [_sample("s1", source=Source.BIRD, difficulty="simple")]
        with pytest.raises(ValueError, match=r"not Source\.MINIDEV_CRUD"):
            inject_crud_samples([], wrong, count=1)
