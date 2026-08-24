"""Unit tests for train/experiments/peft_method_comparison.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelhub.common.capability import CheckStatus
from modelhub.train.experiments.peft_method_comparison import (
    PeftMethod,
    PeftMethodExperimentConfig,
    check_llamafactory_available,
    num_steps_for_method,
    render_peft_method_yaml,
    run_peft_method_comparison_group,
    trainable_param_count_for_method,
    write_peft_method_config,
)


def _config(**overrides: object) -> PeftMethodExperimentConfig:
    defaults: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "dataset_name": "bird23_train_filtered_spider_train",
        "template": "qwen",
        "cutoff_len": 4096,
        "seed": 42,
        "lora_rank": 32,
        "lora_target_modules": ["gate_proj", "q_proj", "in_proj_qkvz"],
        "qlora_bits": 4,
        "zero3_config_relpath": "artifacts/ds_zero3.json",
        "lora_num_steps": 300,
        "full_num_steps": 200,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "output_root": "artifacts/train/experiments/peft_method",
    }
    defaults.update(overrides)
    return PeftMethodExperimentConfig.model_validate(defaults)


class TestNumStepsForMethod:
    def test_lora_uses_lora_num_steps(self) -> None:
        assert num_steps_for_method(PeftMethod.LORA, _config()) == 300

    def test_qlora_uses_lora_num_steps(self) -> None:
        assert num_steps_for_method(PeftMethod.QLORA, _config()) == 300

    def test_full_zero3_uses_full_num_steps(self) -> None:
        assert num_steps_for_method(PeftMethod.FULL_ZERO3, _config()) == 200


class TestTrainableParamCountForMethod:
    def test_lora_returns_adapter_count(self) -> None:
        count = trainable_param_count_for_method(
            PeftMethod.LORA, total_param_count=9_000_000_000, lora_trainable_param_count=50_000_000
        )
        assert count == 50_000_000

    def test_qlora_returns_same_adapter_count_as_lora(self) -> None:
        count = trainable_param_count_for_method(
            PeftMethod.QLORA,
            total_param_count=9_000_000_000,
            lora_trainable_param_count=50_000_000,
        )
        assert count == 50_000_000

    def test_full_zero3_returns_total_count(self) -> None:
        count = trainable_param_count_for_method(
            PeftMethod.FULL_ZERO3,
            total_param_count=9_000_000_000,
            lora_trainable_param_count=50_000_000,
        )
        assert count == 9_000_000_000

    def test_non_positive_total_param_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_param_count"):
            trainable_param_count_for_method(
                PeftMethod.LORA, total_param_count=0, lora_trainable_param_count=1
            )


class TestRenderPeftMethodYaml:
    def test_lora_has_no_quantization_keys(self) -> None:
        payload = yaml.safe_load(render_peft_method_yaml(PeftMethod.LORA, _config()))
        assert payload["finetuning_type"] == "lora"
        assert "quantization_bit" not in payload
        assert payload["max_steps"] == 300

    def test_qlora_has_real_bnb_quantization_keys(self) -> None:
        payload = yaml.safe_load(render_peft_method_yaml(PeftMethod.QLORA, _config()))
        assert payload["quantization_bit"] == 4
        assert payload["quantization_method"] == "bnb"

    def test_full_zero3_uses_deepspeed_key(self) -> None:
        payload = yaml.safe_load(render_peft_method_yaml(PeftMethod.FULL_ZERO3, _config()))
        assert payload["finetuning_type"] == "full"
        assert payload["deepspeed"] == "artifacts/ds_zero3.json"
        assert payload["max_steps"] == 200
        assert "lora_rank" not in payload

    def test_output_dir_differs_per_method(self) -> None:
        lora_payload = yaml.safe_load(render_peft_method_yaml(PeftMethod.LORA, _config()))
        qlora_payload = yaml.safe_load(render_peft_method_yaml(PeftMethod.QLORA, _config()))
        assert lora_payload["output_dir"] != qlora_payload["output_dir"]


class TestWritePeftMethodConfig:
    def test_writes_a_loadable_yaml_file(self, tmp_path: Path) -> None:
        path = write_peft_method_config(PeftMethod.LORA, _config(), tmp_path / "lora.yaml")
        assert path.is_file()
        payload = yaml.safe_load(path.read_text())
        assert payload["finetuning_type"] == "lora"


class TestCheckLlamaFactoryAvailable:
    def test_reports_a_real_check_status(self) -> None:
        result = check_llamafactory_available()
        assert result.status in {CheckStatus.PASS, CheckStatus.SKIP}


class TestRunPeftMethodComparisonGroup:
    def test_raises_when_llamafactory_not_available(self, tmp_path: Path) -> None:
        availability = check_llamafactory_available()
        if availability.status is CheckStatus.PASS:
            pytest.skip("llamafactory-cli is actually installed in this environment")
        with pytest.raises(RuntimeError, match="cannot run peft-method comparison group"):
            run_peft_method_comparison_group(
                PeftMethod.LORA,
                config=_config(),
                yaml_config_path=tmp_path / "lora.yaml",
                experiment_metrics_path=tmp_path / "experiment_metrics.json",
                gpu_cost_per_hour=2.0,
            )
