"""Unit tests for train/runner.py — `run_sft_preflight`'s sequential
go/no-go checks, and the llamafactory-cli subprocess wiring.

This sandbox has no CUDA device, so a *real* `detect_gdn_kernel_status()`
call is always `degraded=True` — the one dedicated test for that
specific gate relies on that real, honest degraded result rather than
faking it. Every other preflight test monkeypatches
`detect_gdn_kernel_status` to a real, successful `KernelStatus` so it
can exercise the checks that run around it without the sandbox's
missing-CUDA fact drowning out what's actually being tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.common.capability import CheckStatus
from modelhub.common.run_manifest import KernelStatus
from modelhub.train import runner
from modelhub.train.dry_run import MemoryEstimateConfig
from modelhub.train.gdn_lora_coverage import GdnLoraCoverageConfig
from modelhub.train.sft_config import (
    SftConfig,
    SftDatasetConfig,
    SftLoraConfig,
    default_qwen35_gdn_target_modules,
)

_TOTAL_LAYERS = 32
_FULL_ATTENTION_LAYER_INDICES = {0, 4, 8, 12, 16, 20, 24, 28}


def _fake_qwen35_module_names() -> list[str]:
    names: list[str] = []
    for i in range(_TOTAL_LAYERS):
        prefix = f"model.layers.{i}"
        names += [f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj", f"{prefix}.mlp.down_proj"]
        if i in _FULL_ATTENTION_LAYER_INDICES:
            names += [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ]
        else:
            names += [
                f"{prefix}.linear_attn.in_proj_qkvz",
                f"{prefix}.linear_attn.in_proj_ba",
                f"{prefix}.linear_attn.out_proj",
                f"{prefix}.linear_attn.conv1d",
            ]
    return names


def _valid_record(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "db_id": "school",
        "question": "how many students?",
        "evidence": None,
        "gold_sql": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
        "source": "bird",
        "split": "train",
        "dialect": "sqlite",
    }


def _sft_config(**overrides: object) -> SftConfig:
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


def _coverage_config(**overrides: object) -> GdnLoraCoverageConfig:
    defaults: dict[str, object] = {
        "total_layers": _TOTAL_LAYERS,
        "target_modules": default_qwen35_gdn_target_modules(),
    }
    defaults.update(overrides)
    return GdnLoraCoverageConfig.model_validate(defaults)


def _memory_config(**overrides: object) -> MemoryEstimateConfig:
    defaults: dict[str, object] = {
        "gpu_memory_bytes": 85899345920,
        "gpu_memory_utilization": 0.9,
        "base_model_weight_bytes": 19327352832,
        "lora_trainable_param_bytes": 209715200,
        "optimizer_state_multiplier": 2.0,
        "estimated_activation_bytes_per_token": 524288,
        "fixed_overhead_bytes": 2684354560,
    }
    defaults.update(overrides)
    return MemoryEstimateConfig.model_validate(defaults)


@pytest.fixture
def healthy_kernel_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "detect_gdn_kernel_status",
        lambda: KernelStatus(causal_conv1d=True, fla=True, degraded=False),
    )


class TestRunSftPreflight:
    def test_healthy_run_succeeds(self, healthy_kernel_status: None) -> None:
        records = [_valid_record(f"s{i}") for i in range(10)]
        report = runner.run_sft_preflight(
            raw_dataset_records=records,
            token_counts=[500] * 10,
            model_named_modules=_fake_qwen35_module_names(),
            sft_config=_sft_config(),
            coverage_config=_coverage_config(),
            memory_config=_memory_config(),
        )
        assert report.dataset_size == 10
        assert report.coverage_report.fully_covered
        assert report.memory_headroom_pct > 0

    def test_bad_schema_hard_fails(self, healthy_kernel_status: None) -> None:
        bad = _valid_record("s1")
        del bad["gold_sql"]
        with pytest.raises(ValueError, match="index 0 failed schema validation"):
            runner.run_sft_preflight(
                raw_dataset_records=[bad],
                token_counts=[500],
                model_named_modules=_fake_qwen35_module_names(),
                sft_config=_sft_config(),
                coverage_config=_coverage_config(),
                memory_config=_memory_config(),
            )

    def test_token_counts_length_mismatch_raises(self, healthy_kernel_status: None) -> None:
        records = [_valid_record(f"s{i}") for i in range(3)]
        with pytest.raises(ValueError, match="does not match"):
            runner.run_sft_preflight(
                raw_dataset_records=records,
                token_counts=[500, 500],  # only 2, but 3 records
                model_named_modules=_fake_qwen35_module_names(),
                sft_config=_sft_config(),
                coverage_config=_coverage_config(),
                memory_config=_memory_config(),
            )

    def test_naive_target_modules_fails_coverage_gate(self, healthy_kernel_status: None) -> None:
        records = [_valid_record("s0")]
        naive_coverage_config = _coverage_config(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        with pytest.raises(ValueError, match="leaves 24/32 layers completely untrained"):
            runner.run_sft_preflight(
                raw_dataset_records=records,
                token_counts=[500],
                model_named_modules=_fake_qwen35_module_names(),
                sft_config=_sft_config(),
                coverage_config=naive_coverage_config,
                memory_config=_memory_config(),
            )

    def test_degraded_kernel_status_refuses_to_start(self) -> None:
        # no monkeypatch: this sandbox has no CUDA device, so the real
        # detect_gdn_kernel_status() is honestly degraded=True here.
        records = [_valid_record("s0")]
        with pytest.raises(ValueError, match="GDN fast-kernel status is degraded"):
            runner.run_sft_preflight(
                raw_dataset_records=records,
                token_counts=[500],
                model_named_modules=_fake_qwen35_module_names(),
                sft_config=_sft_config(),
                coverage_config=_coverage_config(),
                memory_config=_memory_config(),
            )

    def test_oom_estimate_refuses_to_start(self, healthy_kernel_status: None) -> None:
        records = [_valid_record("s0")]
        with pytest.raises(ValueError, match="exceeds the usable budget"):
            runner.run_sft_preflight(
                raw_dataset_records=records,
                token_counts=[500],
                model_named_modules=_fake_qwen35_module_names(),
                sft_config=_sft_config(batch_size=1_000_000),
                coverage_config=_coverage_config(),
                memory_config=_memory_config(),
            )


class TestCheckLlamaFactoryAvailable:
    def test_reports_skip_when_not_on_path(self) -> None:
        result = runner.check_llamafactory_available()
        assert result.status in {CheckStatus.PASS, CheckStatus.SKIP}


class TestBuildTrainingCommand:
    def test_builds_the_real_cli_invocation(self, tmp_path: Path) -> None:
        command = runner.build_training_command(tmp_path / "sft.yaml")
        assert command[0] == "llamafactory-cli"
        assert command[1] == "train"
        assert command[2] == str(tmp_path / "sft.yaml")


class TestRunSftTraining:
    def test_raises_when_llamafactory_not_available(self, tmp_path: Path) -> None:
        availability = runner.check_llamafactory_available()
        if availability.status is CheckStatus.PASS:
            pytest.skip("llamafactory-cli is actually installed in this environment")
        with pytest.raises(RuntimeError, match="cannot start SFT training"):
            runner.run_sft_training(_sft_config(), config_path=tmp_path / "sft.yaml")
