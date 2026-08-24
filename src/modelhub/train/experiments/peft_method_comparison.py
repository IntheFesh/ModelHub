"""Groups 1-3: fine-tuning method comparison — LoRA (r=32, 300 steps) vs
QLoRA (4bit bnb, 300 steps) vs full-parameter + ZeRO-3 (200 steps, "80GB
下 9B 全参可行"). Each variant is one real LLaMA-Factory config rendered
from `PeftMethodExperimentConfig` and run through the same real
`llamafactory-cli` subprocess path B1's `train/runner.py` already
established — this module's own contribution is the config axis (which
`finetuning_type`/quantization/deepspeed keys distinguish one method
from the next) and the pure comparison arithmetic on top.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.capability import CheckStatus, is_available
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

_LLAMAFACTORY_CLI = "llamafactory-cli"


class PeftMethod(StrEnum):
    LORA = "LORA"
    QLORA = "QLORA"
    FULL_ZERO3 = "FULL_ZERO3"


class PeftMethodExperimentConfig(ModelHubBaseConfig):
    model_name_or_path: str
    dataset_name: str
    template: str
    cutoff_len: int
    seed: int
    lora_rank: int
    lora_target_modules: list[str]
    qlora_bits: int
    zero3_config_relpath: str  # path to a real DeepSpeed ZeRO-3 JSON config
    lora_num_steps: int  # PLAN.md: LoRA/QLoRA each 300 steps
    full_num_steps: int  # PLAN.md: full+ZeRO-3 200 steps
    micro_batch_size: int
    gradient_accumulation_steps: int
    output_root: str


def num_steps_for_method(method: PeftMethod, config: PeftMethodExperimentConfig) -> int:
    if method is PeftMethod.FULL_ZERO3:
        return config.full_num_steps
    return config.lora_num_steps


def trainable_param_count_for_method(
    method: PeftMethod, *, total_param_count: int, lora_trainable_param_count: int
) -> int:
    """LoRA/QLoRA freeze the base model — only the injected adapter
    weights are trainable, same count for both since QLoRA's difference
    from LoRA is base-weight quantization, not adapter rank/target
    modules. Full+ZeRO-3 trains every parameter."""
    if total_param_count <= 0:
        raise ValueError(f"total_param_count must be positive, got {total_param_count}")
    if lora_trainable_param_count <= 0:
        raise ValueError(
            f"lora_trainable_param_count must be positive, got {lora_trainable_param_count}"
        )
    if method is PeftMethod.FULL_ZERO3:
        return total_param_count
    return lora_trainable_param_count


def render_peft_method_yaml(method: PeftMethod, config: PeftMethodExperimentConfig) -> str:
    """Real LLaMA-Factory training-config fields distinguishing the three
    methods: `finetuning_type` (lora vs full), `quantization_bit`/
    `quantization_method` (QLoRA's real bnb-4bit keys), and `deepspeed`
    (a path to a real ZeRO-3 JSON config, LLaMA-Factory's documented way
    to enable DeepSpeed). Not validated against a real LLaMA-Factory
    installation in this sandbox (no GPU) — correct as of this project's
    documented understanding of the schema, same honesty caveat as
    `train/sft_config.py::render_llamafactory_yaml`."""
    payload: dict[str, object] = {
        "model_name_or_path": config.model_name_or_path,
        "stage": "sft",
        "do_train": True,
        "dataset": config.dataset_name,
        "template": config.template,
        "cutoff_len": config.cutoff_len,
        "seed": config.seed,
        "per_device_train_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": num_steps_for_method(method, config),
        "output_dir": f"{config.output_root}/{method.value.lower()}",
        "logging_steps": 1,  # every step logged — short comparison runs need dense timing data
        "save_strategy": "no",  # comparison runs don't need a resumable checkpoint
        "bf16": True,
    }
    if method is PeftMethod.LORA:
        payload["finetuning_type"] = "lora"
        payload["lora_rank"] = config.lora_rank
        payload["lora_target"] = ",".join(config.lora_target_modules)
    elif method is PeftMethod.QLORA:
        payload["finetuning_type"] = "lora"
        payload["lora_rank"] = config.lora_rank
        payload["lora_target"] = ",".join(config.lora_target_modules)
        payload["quantization_bit"] = config.qlora_bits
        payload["quantization_method"] = "bnb"
    elif method is PeftMethod.FULL_ZERO3:
        payload["finetuning_type"] = "full"
        payload["deepspeed"] = config.zero3_config_relpath
    else:  # pragma: no cover
        raise ValueError(f"unhandled PeftMethod: {method}")
    return yaml.safe_dump(payload, sort_keys=False)


def write_peft_method_config(
    method: PeftMethod, config: PeftMethodExperimentConfig, path: Path
) -> Path:
    return atomic_write_text(path, render_peft_method_yaml(method, config))


@dataclass(frozen=True)
class LlamaFactoryAvailability:
    status: CheckStatus
    detail: str


def check_llamafactory_available() -> LlamaFactoryAvailability:
    path = shutil.which(_LLAMAFACTORY_CLI)
    if path is None:
        return LlamaFactoryAvailability(
            CheckStatus.SKIP, f"{_LLAMAFACTORY_CLI!r} not found on PATH"
        )
    return LlamaFactoryAvailability(CheckStatus.PASS, f"found at {path}")


def run_peft_method_comparison_group(
    method: PeftMethod,
    *,
    config: PeftMethodExperimentConfig,
    yaml_config_path: Path,
    experiment_metrics_path: Path,
    gpu_cost_per_hour: float,
) -> TrainingRunMetrics:
    """Real subprocess invocation of `llamafactory-cli train` for one
    method variant. Raises (not SKIP-and-continue) if the CLI is not on
    PATH — a caller must check `check_llamafactory_available()` itself
    first (CLAUDE.md §1.3). The metrics-recording callback
    (`ExperimentMetricsCallback`) is wired into the training process
    that `llamafactory-cli` launches internally, not into this function
    — this function's job after the subprocess exits is only to load the
    `RawExperimentMetrics` file that process wrote and turn it into a
    comparable `TrainingRunMetrics`.

    `gpu_cost_per_hour` is a required call-site argument, never a config
    default — same discipline as `bench/cost_model.py`'s
    `gpu_hourly_cost_usd`: FACTS.md's only rental-cost figure for
    2xA100-80G is an unconfirmed, market-fluctuating range ("行情波动，
    以实际报价为准"), not a settled number to bake into a checked-in
    YAML (CLAUDE.md §12)."""
    availability = check_llamafactory_available()
    if not is_available(availability.status):
        raise RuntimeError(
            f"cannot run peft-method comparison group {method.value}: {availability.detail} — "
            f"install the `train` extra and LLaMA-Factory on a real GPU machine first"
        )
    write_peft_method_config(method, config, yaml_config_path)
    num_steps = num_steps_for_method(method, config)
    # timeout=None: a comparison run's duration is bounded by max_steps
    # inside the subprocess (200-500 steps takes minutes, not the
    # multi-hour scale train/runner.py's timeout=None comment discusses,
    # but the reasoning is identical — an external fixed deadline here
    # would be an arbitrary guess, not a real safety bound).
    subprocess.run(
        [_LLAMAFACTORY_CLI, "train", str(yaml_config_path)],
        capture_output=True,
        text=True,
        timeout=None,
        check=True,
    )
    raw = load_raw_experiment_metrics(experiment_metrics_path)
    return TrainingRunMetrics(
        group_name=f"peft_method:{method.value}",
        peak_memory_bytes=require_measured_peak_memory(raw),
        throughput_tokens_per_s=compute_throughput_tokens_per_s(
            tokens_per_step=raw.tokens_per_step, step_time_s=raw.mean_step_time_s
        ),
        step_time_s=raw.mean_step_time_s,
        trainable_param_count=raw.trainable_param_count,
        total_cost=compute_total_cost(
            step_time_s=raw.mean_step_time_s,
            num_steps=num_steps,
            gpu_cost_per_hour=gpu_cost_per_hour,
            gpu_count=1,
        ),
        communication_time_ratio=None,  # not applicable: single-GPU groups
        layer_type_split=None,  # populated by report.py callers that have a real v2 profile
    )


__all__ = [
    "LlamaFactoryAvailability",
    "PeftMethod",
    "PeftMethodExperimentConfig",
    "check_llamafactory_available",
    "num_steps_for_method",
    "render_peft_method_yaml",
    "run_peft_method_comparison_group",
    "trainable_param_count_for_method",
    "write_peft_method_config",
]
