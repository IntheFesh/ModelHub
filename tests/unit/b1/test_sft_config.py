"""Unit tests for train/sft_config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelhub.train.sft_config import (
    SftConfig,
    SftDatasetConfig,
    SftLoraConfig,
    default_qwen35_gdn_target_modules,
    render_llamafactory_yaml,
    write_llamafactory_config,
)


def _config(**overrides: object) -> SftConfig:
    defaults: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "output_dir": "artifacts/train/sft-run",
        "dataset": SftDatasetConfig(
            dataset_name="bird23_train_filtered", template="qwen", cutoff_len=4096
        ),
        "lora": SftLoraConfig(
            rank=32, alpha=64, dropout=0.05, target_modules=default_qwen35_gdn_target_modules()
        ),
        "lr": 0.0002,
        "batch_size": 4,
        "seed": 42,
        "max_seq_len": 4096,
        "max_hours": 8.0,
        "checkpoint_interval_minutes": 30.0,
        "logging_steps": 10,
        "thinking_mode": False,
    }
    defaults.update(overrides)
    return SftConfig.model_validate(defaults)


class TestSftConfigValidation:
    def test_valid_config_constructs(self) -> None:
        config = _config()
        assert config.thinking_mode is False

    def test_thinking_mode_true_rejected(self) -> None:
        with pytest.raises(ValueError, match="thinking_mode must be False"):
            _config(thinking_mode=True)

    def test_non_positive_max_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_hours must be positive"):
            _config(max_hours=0.0)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            _config(bogus_field="nope")


class TestDefaultQwen35GdnTargetModules:
    def test_covers_ffn_attention_and_gdn_suffixes(self) -> None:
        modules = default_qwen35_gdn_target_modules()
        for expected in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
            "conv1d",
        ):
            assert expected in modules

    def test_no_duplicates(self) -> None:
        modules = default_qwen35_gdn_target_modules()
        assert len(modules) == len(set(modules))


class TestRenderLlamaFactoryYaml:
    def test_renders_real_llamafactory_fields(self) -> None:
        rendered = render_llamafactory_yaml(_config())
        payload = yaml.safe_load(rendered)
        assert payload["model_name_or_path"] == "Qwen/Qwen3.5-9B"
        assert payload["stage"] == "sft"
        assert payload["finetuning_type"] == "lora"
        assert payload["lora_rank"] == 32
        assert payload["enable_thinking"] is False

    def test_lora_target_is_comma_joined(self) -> None:
        rendered = render_llamafactory_yaml(_config())
        payload = yaml.safe_load(rendered)
        assert "gate_proj" in payload["lora_target"].split(",")


class TestWriteLlamaFactoryConfig:
    def test_writes_a_loadable_yaml_file(self, tmp_path: Path) -> None:
        path = write_llamafactory_config(_config(), tmp_path / "sft.yaml")
        assert path.is_file()
        payload = yaml.safe_load(path.read_text())
        assert payload["stage"] == "sft"
