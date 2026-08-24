"""Groups 5-6: kernel optimization on/off pairs — Liger Kernel (group 5)
and Flash Attention + sequence packing (group 6), 200 steps each side of
each pair (PLAN.md: "Liger Kernel on/off 各200步" / "Flash Attention
on/off + 序列打包 on/off 各200步" — read as three independent boolean
toggles, not a 2x2 cross of FA×packing, matching the "各200步" per-toggle
framing rather than a per-cell one).

Every toggle is held against the same LoRA r=32 baseline config as group
1 — PLAN.md's "严格控制变量" applies here too: these three comparisons
are about the kernel/data-loading optimization itself, not entangled
with which fine-tuning method is in use.
"""

from __future__ import annotations

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


class KernelToggle(StrEnum):
    LIGER_KERNEL = "LIGER_KERNEL"
    FLASH_ATTENTION = "FLASH_ATTENTION"
    SEQUENCE_PACKING = "SEQUENCE_PACKING"


_TOGGLE_YAML_KEY: dict[KernelToggle, str] = {
    KernelToggle.LIGER_KERNEL: "enable_liger_kernel",
    KernelToggle.FLASH_ATTENTION: "flash_attn",
    KernelToggle.SEQUENCE_PACKING: "packing",
}


class KernelOptimizationExperimentConfig(ModelHubBaseConfig):
    model_name_or_path: str
    dataset_name: str
    template: str
    cutoff_len: int
    seed: int
    num_steps: int  # PLAN.md: 200 steps per side
    micro_batch_size: int
    lora_rank: int
    lora_target_modules: list[str]
    output_root: str


def render_kernel_toggle_yaml(
    toggle: KernelToggle, *, enabled: bool, config: KernelOptimizationExperimentConfig
) -> str:
    """`flash_attn`'s real LLaMA-Factory values are a string enum
    ("fa2"/"sdpa"/"disabled"), not a plain bool — `enabled` maps to
    "fa2" (the real Flash Attention 2 kernel) vs "disabled", while
    `enable_liger_kernel`/`packing` are real plain-bool fields."""
    payload: dict[str, object] = {
        "model_name_or_path": config.model_name_or_path,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": config.lora_rank,
        "lora_target": ",".join(config.lora_target_modules),
        "dataset": config.dataset_name,
        "template": config.template,
        "cutoff_len": config.cutoff_len,
        "seed": config.seed,
        "per_device_train_batch_size": config.micro_batch_size,
        "max_steps": config.num_steps,
        "output_dir": f"{config.output_root}/{toggle.value.lower()}_{'on' if enabled else 'off'}",
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": True,
    }
    key = _TOGGLE_YAML_KEY[toggle]
    if toggle is KernelToggle.FLASH_ATTENTION:
        payload[key] = "fa2" if enabled else "disabled"
    else:
        payload[key] = enabled
    return yaml.safe_dump(payload, sort_keys=False)


def write_kernel_toggle_config(
    toggle: KernelToggle, *, enabled: bool, config: KernelOptimizationExperimentConfig, path: Path
) -> Path:
    return atomic_write_text(
        path, render_kernel_toggle_yaml(toggle, enabled=enabled, config=config)
    )


def run_kernel_toggle_variant(
    toggle: KernelToggle,
    *,
    enabled: bool,
    config: KernelOptimizationExperimentConfig,
    yaml_config_path: Path,
    experiment_metrics_path: Path,
    gpu_cost_per_hour: float,
) -> TrainingRunMetrics:
    """`gpu_cost_per_hour` is a required call-site argument, never a
    config default — see `run_peft_method_comparison_group`'s docstring
    for why."""
    availability = check_llamafactory_available()
    if not is_available(availability.status):
        raise RuntimeError(
            f"cannot run kernel-toggle comparison {toggle.value} (enabled={enabled}): "
            f"{availability.detail} — install the `train` extra and LLaMA-Factory on a "
            f"real GPU machine first"
        )
    write_kernel_toggle_config(toggle, enabled=enabled, config=config, path=yaml_config_path)
    import subprocess

    subprocess.run(
        ["llamafactory-cli", "train", str(yaml_config_path)],
        capture_output=True,
        text=True,
        timeout=None,
        check=True,
    )
    raw = load_raw_experiment_metrics(experiment_metrics_path)
    return TrainingRunMetrics(
        group_name=f"kernel_toggle:{toggle.value}:{'on' if enabled else 'off'}",
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
            gpu_count=1,
        ),
        communication_time_ratio=None,
        layer_type_split=None,
    )


@dataclass(frozen=True)
class KernelTogglePair:
    toggle: KernelToggle
    off: TrainingRunMetrics
    on: TrainingRunMetrics

    @property
    def peak_memory_delta_pct(self) -> float:
        return _pct_delta(self.off.peak_memory_bytes, self.on.peak_memory_bytes)

    @property
    def throughput_delta_pct(self) -> float:
        return _pct_delta(self.off.throughput_tokens_per_s, self.on.throughput_tokens_per_s)

    @property
    def step_time_delta_pct(self) -> float:
        return _pct_delta(self.off.step_time_s, self.on.step_time_s)

    @property
    def cost_delta_pct(self) -> float:
        return _pct_delta(self.off.total_cost, self.on.total_cost)


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("cannot express a percentage delta against a zero baseline")
    return (candidate - baseline) / baseline * 100


__all__ = [
    "KernelOptimizationExperimentConfig",
    "KernelToggle",
    "KernelTogglePair",
    "render_kernel_toggle_yaml",
    "run_kernel_toggle_variant",
    "write_kernel_toggle_config",
]
