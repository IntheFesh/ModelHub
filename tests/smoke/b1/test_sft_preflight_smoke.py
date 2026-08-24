"""Smoke test: B1's full pre-flight pipeline end to end at shrunk scale
(CLAUDE.md §1.4 — only the input scale is shrunk to 20 samples; every
real check — schema validation, token stats, GDN coverage assertion,
kernel-status gate, memory estimate — still runs for real).

`detect_gdn_kernel_status` is monkeypatched to a real, successful
result: this sandbox has no CUDA device, so the genuinely-measured
result here is always degraded=True, which would make every run of
this smoke test fail on a gate this test isn't about (that specific
gate already has its own dedicated coverage in test_runner.py and the
meta tests)."""

from __future__ import annotations

import pytest

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
_SMOKE_SAMPLE_COUNT = 20  # shrunk scale, per CLAUDE.md §1.4


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


def test_full_preflight_pipeline_at_smoke_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "detect_gdn_kernel_status",
        lambda: KernelStatus(causal_conv1d=True, fla=True, degraded=False),
    )
    target_modules = default_qwen35_gdn_target_modules()
    sft_config = SftConfig.model_validate(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "output_dir": "artifacts/train/smoke-run",
            "dataset": SftDatasetConfig(
                dataset_name="bird23_train_filtered", template="qwen", cutoff_len=4096
            ),
            "lora": SftLoraConfig(rank=32, alpha=64, dropout=0.05, target_modules=target_modules),
            "lr": 0.0002,
            "batch_size": 4,
            "seed": 42,
            "max_seq_len": 4096,
            "max_hours": 8.0,
            "checkpoint_interval_minutes": 30.0,
            "logging_steps": 10,
            "thinking_mode": False,
        }
    )
    coverage_config = GdnLoraCoverageConfig.model_validate(
        {"total_layers": _TOTAL_LAYERS, "target_modules": target_modules}
    )
    memory_config = MemoryEstimateConfig.model_validate(
        {
            "gpu_memory_bytes": 85899345920,
            "gpu_memory_utilization": 0.9,
            "base_model_weight_bytes": 19327352832,
            "lora_trainable_param_bytes": 209715200,
            "optimizer_state_multiplier": 2.0,
            "estimated_activation_bytes_per_token": 524288,
            "fixed_overhead_bytes": 2684354560,
        }
    )
    records = [_valid_record(f"smoke-{i}") for i in range(_SMOKE_SAMPLE_COUNT)]
    token_counts = [300 + i * 10 for i in range(_SMOKE_SAMPLE_COUNT)]

    report = runner.run_sft_preflight(
        raw_dataset_records=records,
        token_counts=token_counts,
        model_named_modules=_fake_qwen35_module_names(),
        sft_config=sft_config,
        coverage_config=coverage_config,
        memory_config=memory_config,
    )

    assert report.dataset_size == _SMOKE_SAMPLE_COUNT
    assert report.coverage_report.fully_covered
    assert report.token_length_stats.count == _SMOKE_SAMPLE_COUNT
    assert report.memory_headroom_pct > 0
