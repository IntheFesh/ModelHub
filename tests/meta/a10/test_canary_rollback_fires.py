"""PLAN.md's literal drill, in code: "灰度 5% → 注入劣化 → 自动回滚".

A model is deployed and its canary rollout starts at stage 0 (5% traffic,
`configs/release/canary.yaml`'s real first stage). Once enough requests are
routed, a degraded batch of recent canary predictions — error rate far above
`RollbackConfig.max_error_rate` — is fed through `run_canary_cycle`. This
asserts the automatic-rollback decision actually fires: `CycleAction.
ROLLED_BACK`, the rollout state flips to `ROLLED_BACK`, and the registry
entry for that canary version is written back as `ROLLED_BACK` too — not
just a decision object nobody acts on.

Also proves the drill isn't a test that's always red: the same setup with
healthy canary traffic advances the rollout instead.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.a10.fakes import failing_prediction, healthy_prediction

from modelhub.gate.types import GateDecision, GateResult, GateVerdict
from modelhub.release.canary import (
    CanaryConfig,
    CanaryRolloutState,
    RolloutStatus,
    record_canary_request,
    start_rollout,
)
from modelhub.release.orchestrator import CycleAction, run_canary_cycle
from modelhub.release.registry import ModelRegistry, ModelStatus
from modelhub.release.rollback import RollbackConfig

_MODEL_ID = "arctic-text2sql-r1-7b"
_CANARY_VERSION = "v7-candidate"


def _deploy_canary_candidate(tmp_path: Path) -> ModelRegistry:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate(_MODEL_ID, _CANARY_VERSION, "run-abcdef")
    registry.record_gate_verdict(
        _MODEL_ID,
        _CANARY_VERSION,
        GateVerdict((GateResult("accuracy", GateDecision.PASS, "passed admission gate", {}),)),
    )
    registry.mark_deployed(_MODEL_ID, _CANARY_VERSION)
    return registry


def _rollout_at_first_stage() -> tuple[CanaryRolloutState, CanaryConfig]:
    # real config values from configs/release/canary.yaml, not invented
    # thresholds — the drill exercises the actual operational ramp.
    canary_config = CanaryConfig.model_validate(
        {"stages": [5, 25, 50, 100], "min_requests_per_stage": 100}
    )
    state = start_rollout()
    assert state.current_percentage(canary_config) == 5
    for _ in range(100):
        state = record_canary_request(state)
    return state, canary_config


def test_degraded_canary_at_5_percent_triggers_automatic_rollback(tmp_path: Path) -> None:
    registry = _deploy_canary_candidate(tmp_path)
    state, canary_config = _rollout_at_first_stage()
    rollback_config = RollbackConfig.model_validate(
        {
            "min_sample_size": 30,
            "max_error_rate": 0.10,
            "max_harness_error_rate": 0.02,
            "max_latency_p99_s": 5.0,
        }
    )

    # deliberately injected degradation: 80% of recent canary requests
    # failing with a model-side SQL error, far past the 10% rollback bar.
    degraded_traffic = [failing_prediction(f"bad-{i}") for i in range(80)] + [
        healthy_prediction(f"ok-{i}") for i in range(20)
    ]

    result = run_canary_cycle(
        state,
        recent_canary_predictions=degraded_traffic,
        canary_config=canary_config,
        rollback_config=rollback_config,
        sample_size=50,
        seed=42,
        registry=registry,
        model_id=_MODEL_ID,
        canary_version=_CANARY_VERSION,
    )

    assert result.action is CycleAction.ROLLED_BACK
    assert result.state.status is RolloutStatus.ROLLED_BACK
    assert result.health.decision is GateDecision.REJECT
    assert "error_rate" in result.health.detail

    entry = registry.get(_MODEL_ID, _CANARY_VERSION)
    assert entry is not None
    assert entry.status is ModelStatus.ROLLED_BACK
    assert registry.current_deployed(_MODEL_ID) is None


def test_this_drill_is_not_always_red_healthy_canary_advances_instead(tmp_path: Path) -> None:
    registry = _deploy_canary_candidate(tmp_path)
    state, canary_config = _rollout_at_first_stage()
    rollback_config = RollbackConfig.model_validate(
        {
            "min_sample_size": 30,
            "max_error_rate": 0.10,
            "max_harness_error_rate": 0.02,
            "max_latency_p99_s": 5.0,
        }
    )

    healthy_traffic = [healthy_prediction(f"ok-{i}") for i in range(50)]

    result = run_canary_cycle(
        state,
        recent_canary_predictions=healthy_traffic,
        canary_config=canary_config,
        rollback_config=rollback_config,
        sample_size=50,
        seed=42,
        registry=registry,
        model_id=_MODEL_ID,
        canary_version=_CANARY_VERSION,
    )

    assert result.action is CycleAction.ADVANCED
    assert result.state.current_percentage(canary_config) == 25
    entry = registry.get(_MODEL_ID, _CANARY_VERSION)
    assert entry is not None
    assert entry.status is ModelStatus.DEPLOYED
