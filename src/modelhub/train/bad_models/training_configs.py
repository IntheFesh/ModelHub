"""LLaMA-Factory config rendering for B3's two trained-from-scratch
profiles (ckpt-B, ckpt-D — ckpt-A needs no training config at all, see
`checkpoint_selection.py`).

Both are 200-step short runs (PLAN.md's own literal number, same shape
as B2's comparison groups), always LoRA-only — PLAN.md ckpt-A/B/D's
shared constraint: "每个只导出 LoRA adapter...不传合并权重" (see
`adapter_export.py`). Reuses B1's `default_qwen35_gdn_target_modules`
for the same reason B2 does: any LoRA config in this project must cover
all 32 Qwen3.5-9B layers, drill checkpoints included — an accidentally-
narrow target_modules list would make a drill checkpoint train even
less than intended, muddying which gate actually caught it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.config import ModelHubBaseConfig
from modelhub.train.bad_models.profiles import BadModelProfile


class BadModelTrainingConfig(ModelHubBaseConfig):
    model_name_or_path: str
    dataset_name: str
    template: str
    cutoff_len: int
    seed: int
    num_steps: int
    micro_batch_size: int
    lora_rank: int
    lora_target_modules: list[str]
    output_root: str


def render_bad_model_yaml(profile: BadModelProfile, config: BadModelTrainingConfig) -> str:
    if profile is BadModelProfile.CKPT_A_UNDERFIT:
        raise ValueError(
            "CKPT_A_UNDERFIT needs no training config — PLAN.md: 直接取 B1 主跑的早期 "
            "checkpoint, zero additional training. Use checkpoint_selection.py instead."
        )
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
        "output_dir": f"{config.output_root}/{profile.value.lower()}",
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": True,
    }
    return yaml.safe_dump(payload, sort_keys=False)


def write_bad_model_config(
    profile: BadModelProfile, config: BadModelTrainingConfig, path: Path
) -> Path:
    return atomic_write_text(path, render_bad_model_yaml(profile, config))


__all__ = ["BadModelTrainingConfig", "render_bad_model_yaml", "write_bad_model_config"]
