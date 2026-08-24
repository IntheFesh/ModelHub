"""Unit tests for train/experiments/kernel_optimization_comparison.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelhub.common.capability import CheckStatus
from modelhub.train.experiments.common import TrainingRunMetrics
from modelhub.train.experiments.kernel_optimization_comparison import (
    KernelOptimizationExperimentConfig,
    KernelToggle,
    KernelTogglePair,
    render_kernel_toggle_yaml,
    run_kernel_toggle_variant,
    write_kernel_toggle_config,
)
from modelhub.train.experiments.peft_method_comparison import check_llamafactory_available


def _config(**overrides: object) -> KernelOptimizationExperimentConfig:
    defaults: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "dataset_name": "bird23_train_filtered_spider_train",
        "template": "qwen",
        "cutoff_len": 4096,
        "seed": 42,
        "num_steps": 200,
        "micro_batch_size": 4,
        "lora_rank": 32,
        "lora_target_modules": ["gate_proj", "q_proj", "in_proj_qkvz"],
        "output_root": "artifacts/train/experiments/kernel_optimization",
    }
    defaults.update(overrides)
    return KernelOptimizationExperimentConfig.model_validate(defaults)


class TestRenderKernelToggleYaml:
    def test_liger_kernel_on(self) -> None:
        payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.LIGER_KERNEL, enabled=True, config=_config())
        )
        assert payload["enable_liger_kernel"] is True

    def test_liger_kernel_off(self) -> None:
        payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.LIGER_KERNEL, enabled=False, config=_config())
        )
        assert payload["enable_liger_kernel"] is False

    def test_flash_attention_on_maps_to_fa2(self) -> None:
        payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.FLASH_ATTENTION, enabled=True, config=_config())
        )
        assert payload["flash_attn"] == "fa2"

    def test_flash_attention_off_maps_to_disabled(self) -> None:
        payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.FLASH_ATTENTION, enabled=False, config=_config())
        )
        assert payload["flash_attn"] == "disabled"

    def test_sequence_packing_on(self) -> None:
        payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.SEQUENCE_PACKING, enabled=True, config=_config())
        )
        assert payload["packing"] is True

    def test_output_dir_differs_on_vs_off(self) -> None:
        on_payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.LIGER_KERNEL, enabled=True, config=_config())
        )
        off_payload = yaml.safe_load(
            render_kernel_toggle_yaml(KernelToggle.LIGER_KERNEL, enabled=False, config=_config())
        )
        assert on_payload["output_dir"] != off_payload["output_dir"]


class TestWriteKernelToggleConfig:
    def test_writes_a_loadable_yaml_file(self, tmp_path: Path) -> None:
        path = write_kernel_toggle_config(
            KernelToggle.LIGER_KERNEL,
            enabled=True,
            config=_config(),
            path=tmp_path / "liger_on.yaml",
        )
        assert path.is_file()
        payload = yaml.safe_load(path.read_text())
        assert payload["enable_liger_kernel"] is True


class TestRunKernelToggleVariant:
    def test_raises_when_llamafactory_not_available(self, tmp_path: Path) -> None:
        availability = check_llamafactory_available()
        if availability.status is CheckStatus.PASS:
            pytest.skip("llamafactory-cli is actually installed in this environment")
        with pytest.raises(RuntimeError, match="cannot run kernel-toggle comparison"):
            run_kernel_toggle_variant(
                KernelToggle.LIGER_KERNEL,
                enabled=True,
                config=_config(),
                yaml_config_path=tmp_path / "liger_on.yaml",
                experiment_metrics_path=tmp_path / "experiment_metrics.json",
                gpu_cost_per_hour=2.0,
            )


class TestKernelTogglePair:
    def _metrics(self, **overrides: object) -> TrainingRunMetrics:
        defaults: dict[str, object] = {
            "group_name": "test",
            "peak_memory_bytes": 1000,
            "throughput_tokens_per_s": 100.0,
            "step_time_s": 1.0,
            "trainable_param_count": 50_000_000,
            "total_cost": 1.0,
            "communication_time_ratio": None,
            "layer_type_split": None,
        }
        defaults.update(overrides)
        return TrainingRunMetrics(**defaults)  # type: ignore[arg-type]

    def test_deltas_computed_correctly(self) -> None:
        off = self._metrics(peak_memory_bytes=1000, throughput_tokens_per_s=100.0, step_time_s=1.0)
        on = self._metrics(peak_memory_bytes=800, throughput_tokens_per_s=120.0, step_time_s=0.8)
        pair = KernelTogglePair(toggle=KernelToggle.LIGER_KERNEL, off=off, on=on)
        assert pair.peak_memory_delta_pct == pytest.approx(-20.0)
        assert pair.throughput_delta_pct == pytest.approx(20.0)
        assert pair.step_time_delta_pct == pytest.approx(-20.0)

    def test_zero_baseline_raises(self) -> None:
        off = self._metrics(total_cost=0.0)
        on = self._metrics(total_cost=1.0)
        pair = KernelTogglePair(toggle=KernelToggle.LIGER_KERNEL, off=off, on=on)
        with pytest.raises(ValueError, match="zero baseline"):
            _ = pair.cost_delta_pct
