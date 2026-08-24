"""Unit tests for release/online_sampling.py: seeded sampling of recent
production predictions and their conversion to canary health samples."""

from __future__ import annotations

import pytest
from tests.unit.a10.fakes import failing_prediction, harness_error_prediction, healthy_prediction

from modelhub.release.online_sampling import (
    predictions_to_health_samples,
    sample_recent_predictions,
)


class TestSampleRecentPredictions:
    def test_same_seed_gives_same_sample(self) -> None:
        predictions = [healthy_prediction(f"s{i}") for i in range(100)]
        a = sample_recent_predictions(predictions, sample_size=10, seed=42)
        b = sample_recent_predictions(predictions, sample_size=10, seed=42)
        assert [p.sample_id for p in a] == [p.sample_id for p in b]

    def test_different_seed_can_give_different_sample(self) -> None:
        predictions = [healthy_prediction(f"s{i}") for i in range(200)]
        a = sample_recent_predictions(predictions, sample_size=20, seed=1)
        b = sample_recent_predictions(predictions, sample_size=20, seed=2)
        assert [p.sample_id for p in a] != [p.sample_id for p in b]

    def test_sample_size_at_least_population_returns_everything(self) -> None:
        predictions = [healthy_prediction(f"s{i}") for i in range(5)]
        result = sample_recent_predictions(predictions, sample_size=50, seed=1)
        assert len(result) == 5

    def test_negative_sample_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            sample_recent_predictions([healthy_prediction("s0")], sample_size=-1, seed=1)

    def test_zero_sample_size_returns_empty(self) -> None:
        predictions = [healthy_prediction(f"s{i}") for i in range(5)]
        assert sample_recent_predictions(predictions, sample_size=0, seed=1) == []


class TestPredictionsToHealthSamples:
    def test_exec_ok_counts_as_succeeded(self) -> None:
        samples = predictions_to_health_samples([healthy_prediction("s0", elapsed_s=0.25)])
        assert samples[0].succeeded is True
        assert samples[0].is_harness_error is False
        assert samples[0].latency_s == 0.25

    def test_syntax_error_counts_as_not_succeeded_but_not_harness_error(self) -> None:
        samples = predictions_to_health_samples([failing_prediction("s0")])
        assert samples[0].succeeded is False
        assert samples[0].is_harness_error is False

    def test_harness_error_counts_as_both_not_succeeded_and_harness_error(self) -> None:
        samples = predictions_to_health_samples([harness_error_prediction("s0")])
        assert samples[0].succeeded is False
        assert samples[0].is_harness_error is True

    def test_empty_input_returns_empty_output(self) -> None:
        assert predictions_to_health_samples([]) == []

    def test_preserves_order_and_count(self) -> None:
        predictions = [healthy_prediction("s0"), failing_prediction("s1"), healthy_prediction("s2")]
        samples = predictions_to_health_samples(predictions)
        assert [s.succeeded for s in samples] == [True, False, True]
