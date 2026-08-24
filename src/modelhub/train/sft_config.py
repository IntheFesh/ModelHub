"""SFT training configuration and LLaMA-Factory YAML rendering.

PLAN.md B1 item 1: "LLaMA-Factory 配置化封装（不要重写训练循环）" — this
module's job stops at producing a real, valid LLaMA-Factory training
YAML from this project's own typed config; the actual training loop is
LLaMA-Factory's own CLI (`llamafactory-cli train <config.yaml>`),
invoked — not reimplemented — by whatever real GPU-machine entry point
eventually calls it (out of this sandbox's reach — no GPU, no
LLaMA-Factory install here).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.config import ModelHubBaseConfig
from modelhub.train.gdn_lora_coverage import (
    ATTENTION_MODULE_SUFFIXES,
    FFN_MODULE_SUFFIXES,
    GDN_MODULE_SUFFIXES,
)


class SftDatasetConfig(ModelHubBaseConfig):
    dataset_name: str  # LLaMA-Factory dataset registry key (dataset_info.json)
    template: str  # LLaMA-Factory prompt template name
    cutoff_len: int


class SftLoraConfig(ModelHubBaseConfig):
    rank: int
    alpha: int
    dropout: float
    # must cover all 32 Qwen3.5-9B layers — see gdn_lora_coverage.py,
    # this project's B1 hard acceptance criterion.
    target_modules: list[str]


class SftConfig(ModelHubBaseConfig):
    model_name_or_path: str
    output_dir: str
    dataset: SftDatasetConfig
    lora: SftLoraConfig
    lr: float
    batch_size: int
    seed: int
    max_seq_len: int
    max_hours: float
    checkpoint_interval_minutes: float
    logging_steps: int
    # PLAN.md item 10: "关闭 thinking 模式" — required (no default) so
    # every committed config states this explicitly rather than
    # inheriting a silent default; the validator below additionally
    # enforces it, since PLAN.md frames this as a hard requirement, not
    # a tunable preference.
    thinking_mode: bool

    def model_post_init(self, __context: object) -> None:
        if self.thinking_mode:
            raise ValueError(
                "thinking_mode must be False for Text2SQL SFT — known to "
                "significantly raise the OUTPUT_TRUNCATED rate (PLAN.md B1 item "
                "10 / MODEL-SELECTION-FINAL.md)"
            )
        if self.max_hours <= 0:
            raise ValueError(f"max_hours must be positive, got {self.max_hours}")


def default_qwen35_gdn_target_modules() -> list[str]:
    """FFN suffixes alone already touch all 32 layers (every layer has
    an MLP block regardless of attention type — PLAN.md's "target_modules
    包含 FFN...覆盖全部 32 层" trick); the attention/GDN suffixes are
    included on top so the attention-mixing weights themselves are also
    adapted, not just the FFN."""
    return sorted(FFN_MODULE_SUFFIXES | ATTENTION_MODULE_SUFFIXES | GDN_MODULE_SUFFIXES)


def render_llamafactory_yaml(config: SftConfig) -> str:
    """Every key here is a real LLaMA-Factory training-config field
    (`stage`/`finetuning_type`/`lora_target`/... — see LLaMA-Factory's
    own `examples/` configs), not an invented schema."""
    payload = {
        "model_name_or_path": config.model_name_or_path,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "lora_target": ",".join(config.lora.target_modules),
        "dataset": config.dataset.dataset_name,
        "template": config.dataset.template,
        "cutoff_len": config.dataset.cutoff_len,
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.batch_size,
        "learning_rate": config.lr,
        "seed": config.seed,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "plot_loss": True,
        "bf16": True,
        "enable_thinking": config.thinking_mode,
    }
    return yaml.safe_dump(payload, sort_keys=False)


def write_llamafactory_config(config: SftConfig, path: Path) -> Path:
    return atomic_write_text(path, render_llamafactory_yaml(config))


__all__ = [
    "SftConfig",
    "SftDatasetConfig",
    "SftLoraConfig",
    "default_qwen35_gdn_target_modules",
    "render_llamafactory_yaml",
    "write_llamafactory_config",
]
