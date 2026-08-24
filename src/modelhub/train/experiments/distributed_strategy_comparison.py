"""Group 4: distributed strategy comparison — ZeRO-2 vs ZeRO-3 vs FSDP,
dual-card, 200 steps each (PLAN.md: "第4组另记 all-reduce 时间占比").

All three variants train full parameters (this is where sharding
strategy actually matters — same reasoning as group 3's full+ZeRO-3),
same model/dataset/seed as the other five groups.

★ Honesty scoping, two separate items:

1. ZeRO-2/ZeRO-3 are driven through the same real `deepspeed` YAML key
   LLaMA-Factory documents (`peft_method_comparison.py`'s FULL_ZERO3
   path uses the identical mechanism, just DeepSpeed JSON `zero_
   optimization.stage=2` vs `=3`), launched with `FORCE_TORCHRUN=1` +
   `NPROC_PER_NODE=2` — this project's best understanding of LLaMA-
   Factory's documented multi-GPU invocation, unverified in this
   sandbox (no GPU, no LLaMA-Factory install).
2. FSDP's exact invocation shape through LLaMA-Factory's specific CLI
   abstraction (vs. driving Accelerate/HF Trainer directly) is NOT
   something this project can state with the same confidence — rather
   than guess a plausible-looking but unverified command line,
   `run_distributed_strategy_comparison_group` raises
   `NotImplementedError` for `DistributedStrategy.FSDP` (CLAUDE.md:
   "不确定就留 NotImplementedError，不要给能跑但结果是假的实现"), same
   discipline as `bench/experiments/quantization.py`'s GPTQ calibration
   gap. `render_fsdp_accelerate_config` itself IS written for real
   (Accelerate's FSDP config schema is stable and well-documented) so
   the config-rendering half of this variant is not blocked — only the
   "how do I actually invoke it through llamafactory-cli" half is.

`communication_time_ratio` is real-valued only when a caller supplies
one measured externally (e.g. from a real NCCL/DeepSpeed communication
trace) — this module does not itself instrument communication time, so
every run's `TrainingRunMetrics.communication_time_ratio` from this
module alone is `None` (missing, per CLAUDE.md §2: 缺失就是 None，不是
0), never a fabricated placeholder ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.capability import is_available
from modelhub.common.config import ModelHubBaseConfig
from modelhub.train.experiments.common import (
    TrainingRunMetrics,
    compute_throughput_tokens_per_s,
    compute_total_cost,
)
from modelhub.train.experiments.metrics_recorder import (
    load_raw_experiment_metrics,
    require_measured_peak_memory,
)
from modelhub.train.experiments.peft_method_comparison import check_llamafactory_available


class DistributedStrategy(StrEnum):
    ZERO2 = "ZERO2"
    ZERO3 = "ZERO3"
    FSDP = "FSDP"


class DistributedStrategyExperimentConfig(ModelHubBaseConfig):
    model_name_or_path: str
    dataset_name: str
    template: str
    cutoff_len: int
    seed: int
    num_steps: int  # PLAN.md: 200 steps for this group
    micro_batch_size: int
    gpu_count: int  # PLAN.md: "双卡" — fixed at 2 for this group's real config
    output_root: str


def render_deepspeed_zero_config(stage: int) -> str:
    """Real, standard DeepSpeed ZeRO JSON config fields — `stage` is the
    only axis that differs between the ZERO2 and ZERO3 variants here;
    everything else is deliberately left at DeepSpeed's own documented
    defaults rather than tuned, since this comparison's point is the
    sharding stage itself, not a hand-optimized DeepSpeed config."""
    if stage not in (2, 3):
        raise ValueError(f"stage must be 2 or 3, got {stage}")
    config: dict[str, object] = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }
    return json.dumps(config, indent=2, sort_keys=False)


def render_fsdp_accelerate_config(*, gpu_count: int) -> str:
    """Real Accelerate FSDP config schema fields (`distributed_type: FSDP`,
    `fsdp_config.*`) — Accelerate's own documented config format, stable
    across recent releases. Not itself blocked by the invocation-shape
    uncertainty noted in this module's docstring; only the command that
    would launch training through it is."""
    if gpu_count <= 1:
        raise ValueError(f"FSDP requires gpu_count > 1, got {gpu_count}")
    config: dict[str, object] = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "num_processes": gpu_count,
        "mixed_precision": "bf16",
        "fsdp_config": {
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
        },
    }
    return yaml.safe_dump(config, sort_keys=False)


def render_distributed_strategy_yaml(
    strategy: DistributedStrategy, config: DistributedStrategyExperimentConfig
) -> str:
    payload: dict[str, object] = {
        "model_name_or_path": config.model_name_or_path,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": config.dataset_name,
        "template": config.template,
        "cutoff_len": config.cutoff_len,
        "seed": config.seed,
        "per_device_train_batch_size": config.micro_batch_size,
        "max_steps": config.num_steps,
        "output_dir": f"{config.output_root}/{strategy.value.lower()}",
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": True,
    }
    if strategy in (DistributedStrategy.ZERO2, DistributedStrategy.ZERO3):
        payload["deepspeed"] = f"{config.output_root}/ds_{strategy.value.lower()}.json"
    return yaml.safe_dump(payload, sort_keys=False)


def write_distributed_strategy_configs(
    strategy: DistributedStrategy,
    config: DistributedStrategyExperimentConfig,
    *,
    yaml_config_path: Path,
    deepspeed_config_path: Path | None = None,
    fsdp_config_path: Path | None = None,
) -> None:
    atomic_write_text(yaml_config_path, render_distributed_strategy_yaml(strategy, config))
    if strategy is DistributedStrategy.ZERO2:
        if deepspeed_config_path is None:
            raise ValueError("deepspeed_config_path required for ZERO2")
        atomic_write_text(deepspeed_config_path, render_deepspeed_zero_config(2))
    elif strategy is DistributedStrategy.ZERO3:
        if deepspeed_config_path is None:
            raise ValueError("deepspeed_config_path required for ZERO3")
        atomic_write_text(deepspeed_config_path, render_deepspeed_zero_config(3))
    elif strategy is DistributedStrategy.FSDP:
        if fsdp_config_path is None:
            raise ValueError("fsdp_config_path required for FSDP")
        atomic_write_text(
            fsdp_config_path, render_fsdp_accelerate_config(gpu_count=config.gpu_count)
        )


@dataclass(frozen=True)
class DistributedLaunchEnv:
    """The real env vars LLaMA-Factory documents for multi-GPU without a
    separate launcher script — `FORCE_TORCHRUN=1` tells it to wrap
    itself with `torchrun` using `NPROC_PER_NODE` processes."""

    FORCE_TORCHRUN: str
    NPROC_PER_NODE: str


def build_zero_launch_env(config: DistributedStrategyExperimentConfig) -> DistributedLaunchEnv:
    return DistributedLaunchEnv(FORCE_TORCHRUN="1", NPROC_PER_NODE=str(config.gpu_count))


def run_distributed_strategy_comparison_group(
    strategy: DistributedStrategy,
    *,
    config: DistributedStrategyExperimentConfig,
    yaml_config_path: Path,
    deepspeed_config_path: Path | None,
    experiment_metrics_path: Path,
    gpu_cost_per_hour: float,
) -> TrainingRunMetrics:
    """`gpu_cost_per_hour` is a required call-site argument, never a
    config default — see `run_peft_method_comparison_group`'s docstring
    for why (FACTS.md's only 2xA100-80G rental figure is an unconfirmed,
    market-fluctuating range)."""
    if strategy is DistributedStrategy.FSDP:
        raise NotImplementedError(
            "FSDP's real invocation shape through llamafactory-cli's specific CLI "
            "abstraction is not confirmed by this project (see this module's docstring) "
            "— do not call run_distributed_strategy_comparison_group(FSDP) until it is "
            "verified against a real dual-GPU LLaMA-Factory + Accelerate installation"
        )
    availability = check_llamafactory_available()
    if not is_available(availability.status):
        raise RuntimeError(
            f"cannot run distributed-strategy comparison group {strategy.value}: "
            f"{availability.detail} — install the `train` extra and LLaMA-Factory on a "
            f"real dual-GPU machine first"
        )
    write_distributed_strategy_configs(
        strategy,
        config,
        yaml_config_path=yaml_config_path,
        deepspeed_config_path=deepspeed_config_path,
    )
    import os
    import subprocess

    env = {**os.environ, **build_zero_launch_env(config).__dict__}
    # timeout=None: identical reasoning to peft_method_comparison.py —
    # a 200-step comparison run is bounded by max_steps inside the
    # subprocess, not by an external, arbitrary wall-clock deadline.
    subprocess.run(
        ["llamafactory-cli", "train", str(yaml_config_path)],
        capture_output=True,
        text=True,
        timeout=None,
        check=True,
        env=env,
    )
    raw = load_raw_experiment_metrics(experiment_metrics_path)
    return TrainingRunMetrics(
        group_name=f"distributed_strategy:{strategy.value}",
        peak_memory_bytes=require_measured_peak_memory(raw),
        throughput_tokens_per_s=compute_throughput_tokens_per_s(
            tokens_per_step=raw.tokens_per_step, step_time_s=raw.mean_step_time_s
        ),
        step_time_s=raw.mean_step_time_s,
        trainable_param_count=raw.trainable_param_count,
        total_cost=compute_total_cost(
            step_time_s=raw.mean_step_time_s,
            num_steps=config.num_steps,
            gpu_cost_per_hour=gpu_cost_per_hour,
            gpu_count=config.gpu_count,
        ),
        communication_time_ratio=raw.communication_time_ratio,
        layer_type_split=None,
    )


__all__ = [
    "DistributedLaunchEnv",
    "DistributedStrategy",
    "DistributedStrategyExperimentConfig",
    "build_zero_launch_env",
    "render_deepspeed_zero_config",
    "render_distributed_strategy_yaml",
    "render_fsdp_accelerate_config",
    "run_distributed_strategy_comparison_group",
    "write_distributed_strategy_configs",
]
