"""Unit tests for release/rollback.py: canary health summary + the
automatic-rollback decision."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.gate.types import GateDecision
from modelhub.release.registry import ModelRegistry, ModelStatus
from modelhub.release.rollback import (
    CanaryHealthSample,
    RollbackConfig,
    evaluate_canary_health,
    summarize_canary_health,
    trigger_rollback,
)


def _samples(
    *, n: int, error_rate: float = 0.0, harness_rate: float = 0.0, latency_s: float = 0.1
) -> list[CanaryHealthSample]:
    n_error = round(n * error_rate)
    n_harness = round(n * harness_rate)
    samples = []
    for i in range(n):
        samples.append(
            CanaryHealthSample(
                succeeded=i >= n_error,
                is_harness_error=i < n_harness,
                latency_s=latency_s,
            )
        )
    return samples


def _config(**overrides: object) -> RollbackConfig:
    defaults: dict[str, object] = {
        "min_sample_size": 10,
        "max_error_rate": 0.10,
        "max_harness_error_rate": 0.02,
        "max_latency_p99_s": 5.0,
    }
    defaults.update(overrides)
    return RollbackConfig.model_validate(defaults)


class TestSummarizeCanaryHealth:
    def test_empty_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_canary_health([])

    def test_all_healthy(self) -> None:
        report = summarize_canary_health(_samples(n=10))
        assert report.sample_size == 10
        assert report.error_rate == 0.0
        assert report.harness_error_rate == 0.0

    def test_error_rate_computed(self) -> None:
        report = summarize_canary_health(_samples(n=10, error_rate=0.3))
        assert report.error_rate == pytest.approx(0.3)

    def test_p99_latency_is_nearest_rank(self) -> None:
        samples = [CanaryHealthSample(True, False, float(i)) for i in range(1, 101)]
        report = summarize_canary_health(samples)
        assert report.latency_p99_s == 99.0


class TestEvaluateCanaryHealth:
    def test_not_applicable_below_min_sample_size(self) -> None:
        config = _config(min_sample_size=30)
        result = evaluate_canary_health(_samples(n=5), config)
        assert result.decision is GateDecision.NOT_APPLICABLE
        assert result.metrics["sample_size"] == 5

    def test_pass_when_healthy(self) -> None:
        config = _config(min_sample_size=10)
        result = evaluate_canary_health(_samples(n=50), config)
        assert result.decision is GateDecision.PASS

    def test_reject_on_high_error_rate(self) -> None:
        config = _config(min_sample_size=10, max_error_rate=0.05)
        result = evaluate_canary_health(_samples(n=50, error_rate=0.5), config)
        assert result.decision is GateDecision.REJECT
        assert "error_rate" in result.detail

    def test_reject_on_high_harness_error_rate(self) -> None:
        config = _config(min_sample_size=10, max_harness_error_rate=0.01)
        result = evaluate_canary_health(_samples(n=50, harness_rate=0.2), config)
        assert result.decision is GateDecision.REJECT
        assert "harness_error_rate" in result.detail

    def test_reject_on_high_latency(self) -> None:
        config = _config(min_sample_size=10, max_latency_p99_s=1.0)
        result = evaluate_canary_health(_samples(n=50, latency_s=10.0), config)
        assert result.decision is GateDecision.REJECT
        assert "latency_p99_s" in result.detail

    def test_reject_reports_every_violated_threshold(self) -> None:
        config = _config(
            min_sample_size=10,
            max_error_rate=0.01,
            max_harness_error_rate=0.01,
            max_latency_p99_s=1.0,
        )
        result = evaluate_canary_health(
            _samples(n=50, error_rate=0.5, harness_rate=0.5, latency_s=10.0), config
        )
        assert result.decision is GateDecision.REJECT
        assert "error_rate" in result.detail
        assert "harness_error_rate" in result.detail
        assert "latency_p99_s" in result.detail


class TestTriggerRollback:
    def test_rolls_back_a_deployed_model(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry")
        _deploy(registry, "m", "v1")
        entry = trigger_rollback(registry, "m", "v1", reason="error_rate too high")
        assert entry.status is ModelStatus.ROLLED_BACK
        assert entry.notes == "error_rate too high"


def _deploy(registry: ModelRegistry, model_id: str, version: str) -> None:
    from modelhub.gate.types import GateResult, GateVerdict

    registry.register_candidate(model_id, version, "run-1")
    registry.record_gate_verdict(
        model_id,
        version,
        GateVerdict((GateResult("accuracy", GateDecision.PASS, "ok", {}),)),
    )
    registry.mark_deployed(model_id, version)
