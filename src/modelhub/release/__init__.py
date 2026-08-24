"""Model registry, canary rollout, automatic rollback, online sampling.

A9 owns `registry.py`; A10 owns everything else here.

Public API: `ModelRegistry`/`ModelStatus`/`RegistryEntry` (registry.py);
`route_canary_traffic`/`CanaryConfig`/`CanaryRolloutState`/
`RolloutStatus`/`start_rollout`/`record_canary_request`/
`try_advance_stage`/`mark_rolled_back` (canary.py, staged traffic
routing); `CanaryHealthSample`/`RollbackConfig`/`evaluate_canary_health`/
`summarize_canary_health`/`trigger_rollback` (rollback.py, the automatic
rollback decision — PLAN.md's "灰度 5% → 注入劣化 → 自动回滚" drill);
`sample_recent_predictions`/`predictions_to_health_samples`
(online_sampling.py, seeded/reproducible live-traffic sampling —
CLAUDE.md §7); `run_canary_cycle`/`CanaryCycleResult`/`CycleAction`
(orchestrator.py, ties the three together into one rollout cycle).
"""

from modelhub.release.canary import (
    Arm,
    CanaryConfig,
    CanaryRolloutState,
    RolloutStatus,
    mark_rolled_back,
    record_canary_request,
    route_canary_traffic,
    start_rollout,
    try_advance_stage,
)
from modelhub.release.online_sampling import (
    predictions_to_health_samples,
    sample_recent_predictions,
)
from modelhub.release.orchestrator import CanaryCycleResult, CycleAction, run_canary_cycle
from modelhub.release.registry import GateResultRecord, ModelRegistry, ModelStatus, RegistryEntry
from modelhub.release.rollback import (
    CanaryHealthReport,
    CanaryHealthSample,
    RollbackConfig,
    evaluate_canary_health,
    summarize_canary_health,
    trigger_rollback,
)

__all__ = [
    "Arm",
    "CanaryConfig",
    "CanaryCycleResult",
    "CanaryHealthReport",
    "CanaryHealthSample",
    "CanaryRolloutState",
    "CycleAction",
    "GateResultRecord",
    "ModelRegistry",
    "ModelStatus",
    "RegistryEntry",
    "RollbackConfig",
    "RolloutStatus",
    "evaluate_canary_health",
    "mark_rolled_back",
    "predictions_to_health_samples",
    "record_canary_request",
    "route_canary_traffic",
    "run_canary_cycle",
    "sample_recent_predictions",
    "start_rollout",
    "summarize_canary_health",
    "trigger_rollback",
    "try_advance_stage",
]
