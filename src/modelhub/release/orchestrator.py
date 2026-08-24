"""Ties canary.py + rollback.py + online_sampling.py together: one
"cycle" of a running canary rollout — sample recent canary traffic,
judge its health, and either roll back, advance to the next stage, stay
put pending more data, or (at the last stage) complete the rollout.

This is PLAN.md's drill in code: "灰度 5% → 注入劣化 → 自动回滚" is
exactly `run_canary_cycle` being fed a health-sample set whose error
rate has been pushed over `RollbackConfig.max_error_rate`.

★ `state.requests_at_current_stage` is the caller's responsibility to
keep current (via `canary.py::record_canary_request`, once per routed
request) — this module only acts on whatever state it's handed. A
health cycle runs far less often than individual request routing (e.g.
once per N requests or once per monitoring interval), so the two are
deliberately not coupled into a single call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelhub.eval.records import PredictionRecord
from modelhub.gate.types import GateDecision, GateResult
from modelhub.release.canary import (
    CanaryConfig,
    CanaryRolloutState,
    RolloutStatus,
    mark_rolled_back,
    try_advance_stage,
)
from modelhub.release.online_sampling import (
    predictions_to_health_samples,
    sample_recent_predictions,
)
from modelhub.release.registry import ModelRegistry
from modelhub.release.rollback import RollbackConfig, evaluate_canary_health, trigger_rollback


class CycleAction(StrEnum):
    CONTINUE = "CONTINUE"
    ADVANCED = "ADVANCED"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class CanaryCycleResult:
    action: CycleAction
    state: CanaryRolloutState
    health: GateResult


def run_canary_cycle(
    state: CanaryRolloutState,
    *,
    recent_canary_predictions: list[PredictionRecord],
    canary_config: CanaryConfig,
    rollback_config: RollbackConfig,
    sample_size: int,
    seed: int,
    registry: ModelRegistry,
    model_id: str,
    canary_version: str,
) -> CanaryCycleResult:
    if state.status is not RolloutStatus.IN_PROGRESS:
        raise ValueError(f"cannot run a cycle on a rollout with status {state.status.value}")

    sampled = sample_recent_predictions(
        recent_canary_predictions, sample_size=sample_size, seed=seed
    )
    health_samples = predictions_to_health_samples(sampled)
    health = evaluate_canary_health(health_samples, rollback_config)

    if health.decision is GateDecision.REJECT:
        trigger_rollback(registry, model_id, canary_version, reason=health.detail)
        return CanaryCycleResult(CycleAction.ROLLED_BACK, mark_rolled_back(state), health)

    if health.decision is GateDecision.NOT_APPLICABLE:
        return CanaryCycleResult(CycleAction.CONTINUE, state, health)

    advanced_state = try_advance_stage(state, canary_config)
    if advanced_state.status is RolloutStatus.COMPLETED:
        return CanaryCycleResult(CycleAction.COMPLETED, advanced_state, health)
    if advanced_state.stage_index != state.stage_index:
        return CanaryCycleResult(CycleAction.ADVANCED, advanced_state, health)
    return CanaryCycleResult(CycleAction.CONTINUE, advanced_state, health)


__all__ = ["CanaryCycleResult", "CycleAction", "run_canary_cycle"]
