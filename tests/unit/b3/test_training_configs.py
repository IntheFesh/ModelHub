"""Unit tests for train/bad_models/training_configs.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelhub.train.bad_models.profiles import BadModelProfile
from modelhub.train.bad_models.training_configs import (
    BadModelTrainingConfig,
    render_bad_model_yaml,
    write_bad_model_config,
)


def _config(**overrides: object) -> BadModelTrainingConfig:
    defaults: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "dataset_name": "bird23_train_filtered_spider_train_easy_only",
        "template": "qwen",
        "cutoff_len": 4096,
        "seed": 42,
        "num_steps": 200,
        "micro_batch_size": 4,
        "lora_rank": 32,
        "lora_target_modules": ["gate_proj", "q_proj", "in_proj_qkvz"],
        "output_root": "artifacts/bad_models",
    }
    defaults.update(overrides)
    return BadModelTrainingConfig.model_validate(defaults)


class TestRenderBadModelYaml:
    def test_ckpt_a_raises(self) -> None:
        with pytest.raises(ValueError, match="needs no training config"):
            render_bad_model_yaml(BadModelProfile.CKPT_A_UNDERFIT, _config())

    def test_ckpt_b_renders_real_lora_yaml(self) -> None:
        rendered = render_bad_model_yaml(BadModelProfile.CKPT_B_REGRESSION, _config())
        payload = yaml.safe_load(rendered)
        assert payload["finetuning_type"] == "lora"
        assert payload["max_steps"] == 200

    def test_ckpt_d_output_dir_differs_from_ckpt_b(self) -> None:
        b_payload = yaml.safe_load(
            render_bad_model_yaml(BadModelProfile.CKPT_B_REGRESSION, _config())
        )
        d_payload = yaml.safe_load(render_bad_model_yaml(BadModelProfile.CKPT_D_SAFETY, _config()))
        assert b_payload["output_dir"] != d_payload["output_dir"]


class TestWriteBadModelConfig:
    def test_writes_a_loadable_yaml_file(self, tmp_path: Path) -> None:
        path = write_bad_model_config(
            BadModelProfile.CKPT_B_REGRESSION, _config(), tmp_path / "ckpt_b.yaml"
        )
        assert path.is_file()
        payload = yaml.safe_load(path.read_text())
        assert payload["stage"] == "sft"
