"""Smoke test: a full canary rollout lifecycle, shrunk to a handful of
requests per stage (CLAUDE.md §1.4 — only the input scale is shrunk, every
real code path — routing, health evaluation, registry writes — still runs).
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.a10.fakes import healthy_prediction

from modelhub.gate.types import GateDecision, GateResult, GateVerdict
from modelhub.release.canary import (
    CanaryConfig,
    RolloutStatus,
    record_canary_request,
    route_canary_traffic,
    start_rollout,
)
from modelhub.release.orchestrator import CycleAction, run_canary_cycle
from modelhub.release.registry import ModelRegistry, ModelStatus
from modelhub.release.rollback import RollbackConfig


def test_full_healthy_rollout_reaches_completed_and_stays_deployed(tmp_path: Path) -> None:
    model_id, version = "smoke-model", "v1"
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate(model_id, version, "run-smoke")
    registry.record_gate_verdict(
        model_id, version, GateVerdict((GateResult("accuracy", GateDecision.PASS, "ok", {}),))
    )
    registry.mark_deployed(model_id, version)

    canary_config = CanaryConfig.model_validate({"stages": [5, 100], "min_requests_per_stage": 3})
    rollback_config = RollbackConfig.model_validate(
        {
            "min_sample_size": 3,
            "max_error_rate": 0.10,
            "max_harness_error_rate": 0.02,
            "max_latency_p99_s": 5.0,
        }
    )
    healthy_traffic = [healthy_prediction(f"s{i}") for i in range(10)]

    state = start_rollout()
    assert route_canary_traffic("some-request-key", state.current_percentage(canary_config)) in (
        "stable",
        "canary",
    )

    for _ in range(3):
        state = record_canary_request(state)
    result = run_canary_cycle(
        state,
        recent_canary_predictions=healthy_traffic,
        canary_config=canary_config,
        rollback_config=rollback_config,
        sample_size=10,
        seed=1,
        registry=registry,
        model_id=model_id,
        canary_version=version,
    )
    assert result.action is CycleAction.ADVANCED
    assert result.state.current_percentage(canary_config) == 100

    state = result.state
    for _ in range(3):
        state = record_canary_request(state)
    result = run_canary_cycle(
        state,
        recent_canary_predictions=healthy_traffic,
        canary_config=canary_config,
        rollback_config=rollback_config,
        sample_size=10,
        seed=1,
        registry=registry,
        model_id=model_id,
        canary_version=version,
    )
    assert result.action is CycleAction.COMPLETED
    assert result.state.status is RolloutStatus.COMPLETED

    entry = registry.get(model_id, version)
    assert entry is not None
    assert entry.status is ModelStatus.DEPLOYED
