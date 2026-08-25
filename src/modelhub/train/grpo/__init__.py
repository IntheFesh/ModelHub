"""B5: GRPO on veRL — Qwen3.5-9B, FSDP training side + vLLM rollout
(PLAN.md §B5). This package's real weight is reward-quality protection
(masking system errors, degenerate-group detection, per-step abort
thresholds, rollout SQL timing) — see `reward.py`'s module docstring."""

from modelhub.train.grpo.group_diagnostics import GroupDiagnostics, RolloutGroup, diagnose_group
from modelhub.train.grpo.kernel_guard import guard_rollout_kernel_status
from modelhub.train.grpo.reward import (
    REWARD_FUNCTION_VERSION,
    MaskReason,
    RewardOutcome,
    RolloutReward,
    compute_reward,
)
from modelhub.train.grpo.rollout_timing import RolloutTimingBreakdown, execute_rollout_sql_batch
from modelhub.train.grpo.runner import (
    GrpoTrainingConfig,
    check_verl_available,
    run_grpo_preflight,
    run_grpo_training,
)
from modelhub.train.grpo.smoke_test import (
    GrpoSmokeCriteria,
    assert_grpo_smoke_passed,
    check_peak_memory,
)
from modelhub.train.grpo.step_diagnostics import (
    StepDiagnostics,
    assert_harness_error_rate_ok,
    assert_unclassified_rate_ok,
    compute_step_diagnostics,
)

__all__ = [
    "REWARD_FUNCTION_VERSION",
    "GroupDiagnostics",
    "GrpoSmokeCriteria",
    "GrpoTrainingConfig",
    "MaskReason",
    "RewardOutcome",
    "RolloutGroup",
    "RolloutReward",
    "RolloutTimingBreakdown",
    "StepDiagnostics",
    "assert_grpo_smoke_passed",
    "assert_harness_error_rate_ok",
    "assert_unclassified_rate_ok",
    "check_peak_memory",
    "check_verl_available",
    "compute_reward",
    "compute_step_diagnostics",
    "diagnose_group",
    "execute_rollout_sql_batch",
    "guard_rollout_kernel_status",
    "run_grpo_preflight",
    "run_grpo_training",
]
