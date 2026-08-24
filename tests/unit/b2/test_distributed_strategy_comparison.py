"""Unit tests for train/experiments/distributed_strategy_comparison.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from modelhub.common.capability import CheckStatus
from modelhub.train.experiments.distributed_strategy_comparison import (
    DistributedStrategy,
    DistributedStrategyExperimentConfig,
    build_zero_launch_env,
    render_deepspeed_zero_config,
    render_distributed_strategy_yaml,
    render_fsdp_accelerate_config,
    run_distributed_strategy_comparison_group,
    write_distributed_strategy_configs,
)
from modelhub.train.experiments.peft_method_comparison import check_llamafactory_available


def _config(**overrides: object) -> DistributedStrategyExperimentConfig:
    defaults: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "dataset_name": "bird23_train_filtered_spider_train",
        "template": "qwen",
        "cutoff_len": 4096,
        "seed": 42,
        "num_steps": 200,
        "micro_batch_size": 4,
        "gpu_count": 2,
        "output_root": "artifacts/train/experiments/distributed_strategy",
    }
    defaults.update(overrides)
    return DistributedStrategyExperimentConfig.model_validate(defaults)


class TestRenderDeepspeedZeroConfig:
    def test_stage_2(self) -> None:
        config = json.loads(render_deepspeed_zero_config(2))
        assert config["zero_optimization"]["stage"] == 2

    def test_stage_3(self) -> None:
        config = json.loads(render_deepspeed_zero_config(3))
        assert config["zero_optimization"]["stage"] == 3

    def test_invalid_stage_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage must be 2 or 3"):
            render_deepspeed_zero_config(1)


class TestRenderFsdpAccelerateConfig:
    def test_real_accelerate_fsdp_fields(self) -> None:
        config = yaml.safe_load(render_fsdp_accelerate_config(gpu_count=2))
        assert config["distributed_type"] == "FSDP"
        assert config["num_processes"] == 2
        assert config["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"

    def test_single_gpu_rejected(self) -> None:
        with pytest.raises(ValueError, match="gpu_count > 1"):
            render_fsdp_accelerate_config(gpu_count=1)


class TestRenderDistributedStrategyYaml:
    def test_zero_variants_set_finetuning_type_full_and_deepspeed_key(self) -> None:
        rendered = render_distributed_strategy_yaml(DistributedStrategy.ZERO2, _config())
        payload = yaml.safe_load(rendered)
        assert payload["finetuning_type"] == "full"
        assert "ds_zero2.json" in payload["deepspeed"]

    def test_zero3_deepspeed_path_names_zero3(self) -> None:
        rendered = render_distributed_strategy_yaml(DistributedStrategy.ZERO3, _config())
        payload = yaml.safe_load(rendered)
        assert "ds_zero3.json" in payload["deepspeed"]

    def test_fsdp_variant_has_no_deepspeed_key(self) -> None:
        rendered = render_distributed_strategy_yaml(DistributedStrategy.FSDP, _config())
        payload = yaml.safe_load(rendered)
        assert "deepspeed" not in payload


class TestBuildZeroLaunchEnv:
    def test_matches_configured_gpu_count(self) -> None:
        env = build_zero_launch_env(_config(gpu_count=2))
        assert env.FORCE_TORCHRUN == "1"
        assert env.NPROC_PER_NODE == "2"


class TestWriteDistributedStrategyConfigs:
    def test_zero2_writes_both_files(self, tmp_path: Path) -> None:
        write_distributed_strategy_configs(
            DistributedStrategy.ZERO2,
            _config(),
            yaml_config_path=tmp_path / "zero2.yaml",
            deepspeed_config_path=tmp_path / "ds_zero2.json",
        )
        assert (tmp_path / "zero2.yaml").is_file()
        assert (tmp_path / "ds_zero2.json").is_file()

    def test_zero2_missing_deepspeed_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="deepspeed_config_path required"):
            write_distributed_strategy_configs(
                DistributedStrategy.ZERO2, _config(), yaml_config_path=tmp_path / "zero2.yaml"
            )

    def test_fsdp_missing_fsdp_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fsdp_config_path required"):
            write_distributed_strategy_configs(
                DistributedStrategy.FSDP, _config(), yaml_config_path=tmp_path / "fsdp.yaml"
            )

    def test_fsdp_writes_fsdp_config(self, tmp_path: Path) -> None:
        write_distributed_strategy_configs(
            DistributedStrategy.FSDP,
            _config(),
            yaml_config_path=tmp_path / "fsdp.yaml",
            fsdp_config_path=tmp_path / "fsdp_accelerate.yaml",
        )
        assert (tmp_path / "fsdp_accelerate.yaml").is_file()


class TestRunDistributedStrategyComparisonGroup:
    def test_fsdp_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="FSDP's real invocation shape"):
            run_distributed_strategy_comparison_group(
                DistributedStrategy.FSDP,
                config=_config(),
                yaml_config_path=tmp_path / "fsdp.yaml",
                deepspeed_config_path=None,
                experiment_metrics_path=tmp_path / "experiment_metrics.json",
                gpu_cost_per_hour=2.0,
            )

    def test_zero2_raises_when_llamafactory_not_available(self, tmp_path: Path) -> None:
        availability = check_llamafactory_available()
        if availability.status is CheckStatus.PASS:
            pytest.skip("llamafactory-cli is actually installed in this environment")
        with pytest.raises(RuntimeError, match="cannot run distributed-strategy comparison"):
            run_distributed_strategy_comparison_group(
                DistributedStrategy.ZERO2,
                config=_config(),
                yaml_config_path=tmp_path / "zero2.yaml",
                deepspeed_config_path=tmp_path / "ds_zero2.json",
                experiment_metrics_path=tmp_path / "experiment_metrics.json",
                gpu_cost_per_hour=2.0,
            )
