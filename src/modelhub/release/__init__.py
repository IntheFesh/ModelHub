"""Model registry, canary rollout, automatic rollback, online sampling,
incident log.

A9 owns `registry.py`; A10 owns canary/rollback/online_sampling/
orchestrator; A12 owns `incident_log.py` (a real gap both A9 and A10
left — "拦截记录 append-only 写 docs/incident-log.md" was never actually
wired to a writer until now).

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
(orchestrator.py, ties the three together into one rollout cycle);
`IncidentRecord`/`IncidentType`/`append_incident`/
`gate_rejection_incident`/`canary_rollback_incident`/`read_incident_log`/
`count_incidents` (incident_log.py).
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
from modelhub.release.incident_log import (
    IncidentRecord,
    IncidentType,
    append_incident,
    canary_rollback_incident,
    count_incidents,
    gate_rejection_incident,
    read_incident_log,
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
    "IncidentRecord",
    "IncidentType",
    "ModelRegistry",
    "ModelStatus",
    "RegistryEntry",
    "RollbackConfig",
    "RolloutStatus",
    "append_incident",
    "canary_rollback_incident",
    "count_incidents",
    "evaluate_canary_health",
    "gate_rejection_incident",
    "mark_rolled_back",
    "predictions_to_health_samples",
    "read_incident_log",
    "record_canary_request",
    "route_canary_traffic",
    "run_canary_cycle",
    "sample_recent_predictions",
    "start_rollout",
    "summarize_canary_health",
    "trigger_rollback",
    "try_advance_stage",
]
