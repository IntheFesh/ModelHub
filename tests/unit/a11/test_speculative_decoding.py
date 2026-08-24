"""Unit tests for bench/experiments/speculative_decoding.py."""

from __future__ import annotations

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.experiments.speculative_decoding import (
    ConcurrencySpeedupPoint,
    SpeculativeAcceptanceStats,
    SpeculativeDecayCurve,
    SpeculativeDecodingExperimentConfig,
    SpeculativeDecodingPath,
    measure_concurrency_decay_curve,
)


class TestSpeculativeAcceptanceStats:
    def test_acceptance_rate_computed(self) -> None:
        stats = SpeculativeAcceptanceStats(accepted_tokens=80, draft_tokens=100)
        assert stats.acceptance_rate == pytest.approx(0.8)

    def test_zero_draft_tokens_gives_none_not_zero(self) -> None:
        stats = SpeculativeAcceptanceStats(accepted_tokens=0, draft_tokens=0)
        assert stats.acceptance_rate is None


class TestSpeculativeDecayCurve:
    def test_collapses_true_when_any_point_at_or_below_one(self) -> None:
        curve = SpeculativeDecayCurve(
            path=SpeculativeDecodingPath.MTP,
            points=(
                ConcurrencySpeedupPoint(concurrency=1, baseline_qps=10, speculative_qps=20),
                ConcurrencySpeedupPoint(concurrency=64, baseline_qps=10, speculative_qps=8),
            ),
        )
        assert curve.collapses_under_concurrency is True

    def test_collapses_false_when_every_point_stays_above_one(self) -> None:
        curve = SpeculativeDecayCurve(
            path=SpeculativeDecodingPath.NGRAM,
            points=(
                ConcurrencySpeedupPoint(concurrency=1, baseline_qps=10, speculative_qps=20),
                ConcurrencySpeedupPoint(concurrency=64, baseline_qps=10, speculative_qps=15),
            ),
        )
        assert curve.collapses_under_concurrency is False

    def test_all_zero_baseline_raises(self) -> None:
        curve = SpeculativeDecayCurve(
            path=SpeculativeDecodingPath.MTP,
            points=(ConcurrencySpeedupPoint(concurrency=1, baseline_qps=0, speculative_qps=0),),
        )
        with pytest.raises(ValueError, match="zero valid speedup points"):
            _ = curve.collapses_under_concurrency


class TestMeasureConcurrencyDecayCurve:
    def test_measures_one_point_per_configured_concurrency_level(self) -> None:
        config = SpeculativeDecodingExperimentConfig(
            metrics_url="http://localhost:8000/metrics",
            accepted_tokens_metric_name="vllm:spec_decode_num_accepted_tokens_total",
            draft_tokens_metric_name="vllm:spec_decode_num_draft_tokens_total",
            concurrency_levels=[1, 4, 8],
            num_requests_per_level=5,
            max_tokens=32,
            temperature=0.0,
            generate_timeout_s=5.0,
            prompt_length_label="1k",
        )
        curve = measure_concurrency_decay_curve(
            SpeculativeDecodingPath.MTP,
            baseline_client=BenchFakeClient(),
            speculative_client=BenchFakeClient(),
            prompt="SELECT ...",
            config=config,
        )
        assert [p.concurrency for p in curve.points] == [1, 4, 8]
        assert curve.path == SpeculativeDecodingPath.MTP

    def test_a_slower_speculative_client_produces_a_collapsed_curve(self) -> None:
        config = SpeculativeDecodingExperimentConfig(
            metrics_url="http://localhost:8000/metrics",
            accepted_tokens_metric_name="vllm:spec_decode_num_accepted_tokens_total",
            draft_tokens_metric_name="vllm:spec_decode_num_draft_tokens_total",
            concurrency_levels=[4],
            num_requests_per_level=5,
            max_tokens=32,
            temperature=0.0,
            generate_timeout_s=5.0,
            prompt_length_label="1k",
        )
        curve = measure_concurrency_decay_curve(
            SpeculativeDecodingPath.NGRAM,
            baseline_client=BenchFakeClient(latency_s=0.0),
            speculative_client=BenchFakeClient(latency_s=0.05),
            prompt="SELECT ...",
            config=config,
        )
        assert curve.collapses_under_concurrency is True
