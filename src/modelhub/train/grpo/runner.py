"""B5's top-level GRPO orchestration — PLAN.md: "用 veRL（FSDP 训练侧 +
vLLM rollout）...基座 Qwen3.5-9B...求快可先用同族 2B/4B 验证流水线".

★ Honesty scoping: veRL (the `verl` package) is invoked through a
Hydra/YAML-driven CLI entry point (`python3 -m verl.trainer.main_ppo
<config overrides>`) in every real usage this project could confirm,
not a simple Python class this module can construct and call in-process
the way B1 wraps LLaMA-Factory or B4 wraps TRL's `DPOTrainer`. Rather
than guess an unverified subprocess invocation shape, `run_grpo_training`
raises `NotImplementedError` for the actual launch — same discipline as
`train/experiments/distributed_strategy_comparison.py`'s FSDP gap. Every
piece around it (preflight checks, config rendering, the reward/
diagnostics pipeline in this package's other modules) is real and fully
tested.
"""

from __future__ import annotations

from typing import NoReturn

from modelhub.common.capability import CheckStatus
from modelhub.common.config import ModelHubBaseConfig
from modelhub.train.grpo.kernel_guard import guard_rollout_kernel_status
from modelhub.train.grpo.reward import REWARD_FUNCTION_VERSION

try:
    import verl as _verl
except ImportError:
    _verl = None


class GrpoTrainingConfig(ModelHubBaseConfig):
    base_model_name_or_path: str
    lora_target_modules: list[str]
    rollout_k: int
    rollout_max_concurrency: int
    rollout_timeout_s: float
    harness_error_abort_threshold: float
    seed: int
    max_hours: float
    output_dir: str

    def model_post_init(self, __context: object) -> None:
        if self.rollout_k <= 0:
            raise ValueError(f"rollout_k must be positive, got {self.rollout_k}")
        if self.max_hours <= 0:
            raise ValueError(f"max_hours must be positive, got {self.max_hours}")


def check_verl_available() -> CheckStatus:
    return CheckStatus.PASS if _verl is not None else CheckStatus.SKIP


def run_grpo_preflight(config: GrpoTrainingConfig) -> None:
    """Everything this sandbox CAN check for real before a GRPO run
    starts: rollout-side GDN kernel status (PLAN.md ★ item 9). Does not
    itself check GPU memory or the LoRA coverage requirement — those are
    B1's `gdn_lora_coverage.py`/`dry_run.py` checks, reused unchanged by
    a real GPU-machine entry point rather than duplicated here."""
    guard_rollout_kernel_status()


def run_grpo_training(config: GrpoTrainingConfig) -> NoReturn:
    """Raises (not SKIP-and-continue) if `verl` is not installed — a
    caller must check `check_verl_available()` first (CLAUDE.md §1.3).
    See this module's docstring for why the actual launch is left
    unimplemented rather than guessed."""
    if check_verl_available() is not CheckStatus.PASS:
        raise RuntimeError(
            "cannot start GRPO training: verl is not installed — install it on a real "
            "2xA100-80G machine first"
        )
    run_grpo_preflight(config)
    raise NotImplementedError(
        f"veRL's real launch shape (Hydra/YAML-driven `python -m verl.trainer.main_ppo` "
        f"CLI, not a simple in-process trainer class) is not confirmed by this project — "
        f"wire this up against the real veRL API on first real GPU run rather than "
        f"guessing its exact call shape untested (CLAUDE.md: 不确定就留 NotImplementedError). "
        f"reward_fn_version={REWARD_FUNCTION_VERSION!r} must be written into the real run's "
        f"manifest.reward_fn_version once this is wired up."
    )


__all__ = [
    "GrpoTrainingConfig",
    "check_verl_available",
    "run_grpo_preflight",
    "run_grpo_training",
]
