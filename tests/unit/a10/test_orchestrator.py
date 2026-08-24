"""Unit tests for release/orchestrator.py: one canary-rollout cycle tying
together sampling, health evaluation, stage advancement, and rollback."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a10.fakes import failing_prediction, healthy_prediction

from modelhub.gate.types import GateDecision, GateResult, GateVerdict
from modelhub.release.canary import (
    CanaryConfig,
    RolloutStatus,
    mark_rolled_back,
    record_canary_request,
    start_rollout,
)
from modelhub.release.orchestrator import CycleAction, run_canary_cycle
from modelhub.release.registry import ModelRegistry, ModelStatus
from modelhub.release.rollback import RollbackConfig


def _canary_config(**overrides: object) -> CanaryConfig:
    defaults: dict[str, object] = {"stages": [5, 25, 50, 100], "min_requests_per_stage": 3}
    defaults.update(overrides)
    return CanaryConfig.model_validate(defaults)


def _rollback_config(**overrides: object) -> RollbackConfig:
    defaults: dict[str, object] = {
        "min_sample_size": 10,
        "max_error_rate": 0.10,
        "max_harness_error_rate": 0.02,
        "max_latency_p99_s": 5.0,
    }
    defaults.update(overrides)
    return RollbackConfig.model_validate(defaults)


def _deployed_registry(
    tmp_path: Path, model_id: str = "m", version: str = "canary-v1"
) -> ModelRegistry:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate(model_id, version, "run-1")
    registry.record_gate_verdict(
        model_id, version, GateVerdict((GateResult("accuracy", GateDecision.PASS, "ok", {}),))
    )
    registry.mark_deployed(model_id, version)
    return registry


def test_cycle_raises_if_rollout_not_in_progress(tmp_path: Path) -> None:
    registry = _deployed_registry(tmp_path)
    state = start_rollout()
    for _ in range(10):
        state = record_canary_request(state)
    rolled_back_state = mark_rolled_back(state)
    with pytest.raises(ValueError, match="cannot run a cycle"):
        run_canary_cycle(
            rolled_back_state,
            recent_canary_predictions=[healthy_prediction("s0")],
            canary_config=_canary_config(),
            rollback_config=_rollback_config(),
            sample_size=10,
            seed=1,
            registry=registry,
            model_id="m",
            canary_version="canary-v1",
        )


def test_too_few_samples_continues_without_advancing(tmp_path: Path) -> None:
    registry = _deployed_registry(tmp_path)
    state = start_rollout()
    for _ in range(5):
        state = record_canary_request(state)
    result = run_canary_cycle(
        state,
        recent_canary_predictions=[healthy_prediction(f"s{i}") for i in range(3)],
        canary_config=_canary_config(min_requests_per_stage=3),
        rollback_config=_rollback_config(min_sample_size=30),
        sample_size=30,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert result.action is CycleAction.CONTINUE
    assert result.health.decision is GateDecision.NOT_APPLICABLE
    assert result.state.stage_index == 0


def test_healthy_traffic_advances_stage(tmp_path: Path) -> None:
    registry = _deployed_registry(tmp_path)
    state = start_rollout()
    for _ in range(3):
        state = record_canary_request(state)
    result = run_canary_cycle(
        state,
        recent_canary_predictions=[healthy_prediction(f"s{i}") for i in range(20)],
        canary_config=_canary_config(min_requests_per_stage=3),
        rollback_config=_rollback_config(min_sample_size=10),
        sample_size=20,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert result.action is CycleAction.ADVANCED
    assert result.state.stage_index == 1
    assert result.health.decision is GateDecision.PASS


def test_healthy_traffic_at_last_stage_completes(tmp_path: Path) -> None:
    # completion needs min_requests_per_stage healthy requests *at* the
    # final (100%) stage too, not merely reaching it — so this takes two
    # cycles: one that advances stage 0 -> 1 (the last stage), and a
    # second, after more requests land at 100%, that completes.
    registry = _deployed_registry(tmp_path)
    config = _canary_config(stages=[5, 100], min_requests_per_stage=3)
    healthy_traffic = [healthy_prediction(f"s{i}") for i in range(20)]
    rollback_config = _rollback_config(min_sample_size=10)

    state = start_rollout()
    for _ in range(3):
        state = record_canary_request(state)
    first = run_canary_cycle(
        state,
        recent_canary_predictions=healthy_traffic,
        canary_config=config,
        rollback_config=rollback_config,
        sample_size=20,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert first.action is CycleAction.ADVANCED
    assert first.state.current_percentage(config) == 100

    state = first.state
    for _ in range(3):
        state = record_canary_request(state)
    second = run_canary_cycle(
        state,
        recent_canary_predictions=healthy_traffic,
        canary_config=config,
        rollback_config=rollback_config,
        sample_size=20,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert second.action is CycleAction.COMPLETED
    assert second.state.status is RolloutStatus.COMPLETED


def test_degraded_traffic_triggers_rollback(tmp_path: Path) -> None:
    registry = _deployed_registry(tmp_path)
    state = start_rollout()
    for _ in range(3):
        state = record_canary_request(state)
    degraded_predictions = [failing_prediction(f"s{i}") for i in range(20)] + [
        healthy_prediction(f"h{i}") for i in range(5)
    ]
    result = run_canary_cycle(
        state,
        recent_canary_predictions=degraded_predictions,
        canary_config=_canary_config(min_requests_per_stage=3),
        rollback_config=_rollback_config(min_sample_size=10, max_error_rate=0.10),
        sample_size=25,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert result.action is CycleAction.ROLLED_BACK
    assert result.state.status is RolloutStatus.ROLLED_BACK
    entry = registry.get("m", "canary-v1")
    assert entry is not None
    assert entry.status is ModelStatus.ROLLED_BACK


def test_not_enough_requests_at_stage_stays_at_current_stage_even_when_healthy(
    tmp_path: Path,
) -> None:
    registry = _deployed_registry(tmp_path)
    state = start_rollout()
    state = record_canary_request(state)
    result = run_canary_cycle(
        state,
        recent_canary_predictions=[healthy_prediction(f"s{i}") for i in range(20)],
        canary_config=_canary_config(min_requests_per_stage=10),
        rollback_config=_rollback_config(min_sample_size=10),
        sample_size=20,
        seed=1,
        registry=registry,
        model_id="m",
        canary_version="canary-v1",
    )
    assert result.action is CycleAction.CONTINUE
    assert result.state.stage_index == 0
    assert result.health.decision is GateDecision.PASS
