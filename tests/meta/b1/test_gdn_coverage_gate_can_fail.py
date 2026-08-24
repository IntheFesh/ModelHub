"""CLAUDE.md §1.5 required meta-test: proves B1's hard acceptance
criterion — full 32-layer LoRA coverage — actually rejects the exact
scenario PLAN.md warns about ("target_modules 只写标准 attention 名字，
GDN 层完全没被训练，训练悄悄跑完，没有任何报错").

Injects the naive, attention-only `target_modules` list through the
real `run_sft_preflight` orchestrator (not just the lower-level
`assert_full_layer_coverage` unit) and asserts preflight refuses to
start training, naming the exact uncovered-layer count. Also proves the
gate is not permanently red: the real fix (FFN + attention + GDN
suffixes) passes the same orchestrator end to end.
"""

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


def _sft_config(target_modules: list[str]) -> SftConfig:
    return SftConfig.model_validate(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "output_dir": "artifacts/train/meta-test-run",
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


def _memory_config() -> MemoryEstimateConfig:
    return MemoryEstimateConfig.model_validate(
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


def test_naive_attention_only_target_modules_is_rejected_by_real_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # detect_gdn_kernel_status is monkeypatched to a real, PASSing result
    # so this test isolates the coverage gate specifically — this
    # sandbox's real result (no CUDA device) is degraded=True regardless
    # of target_modules, which would mask what this meta-test is for.
    monkeypatch.setattr(
        runner,
        "detect_gdn_kernel_status",
        lambda: KernelStatus(causal_conv1d=True, fla=True, degraded=False),
    )
    naive_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # PLAN.md's exact bug scenario
    naive_coverage_config = GdnLoraCoverageConfig.model_validate(
        {"total_layers": _TOTAL_LAYERS, "target_modules": naive_target_modules}
    )

    with pytest.raises(ValueError, match="leaves 24/32 layers completely untrained"):
        runner.run_sft_preflight(
            raw_dataset_records=[_valid_record("s0")],
            token_counts=[500],
            model_named_modules=_fake_qwen35_module_names(),
            sft_config=_sft_config(naive_target_modules),
            coverage_config=naive_coverage_config,
            memory_config=_memory_config(),
        )


def test_the_real_fix_passes_the_same_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the gate is not permanently red: the documented fix
    (FFN + attention + GDN suffixes) reaches full coverage through the
    exact same real preflight path the naive config was rejected by."""
    monkeypatch.setattr(
        runner,
        "detect_gdn_kernel_status",
        lambda: KernelStatus(causal_conv1d=True, fla=True, degraded=False),
    )
    fixed_target_modules = default_qwen35_gdn_target_modules()
    fixed_coverage_config = GdnLoraCoverageConfig.model_validate(
        {"total_layers": _TOTAL_LAYERS, "target_modules": fixed_target_modules}
    )

    report = runner.run_sft_preflight(
        raw_dataset_records=[_valid_record("s0")],
        token_counts=[500],
        model_named_modules=_fake_qwen35_module_names(),
        sft_config=_sft_config(fixed_target_modules),
        coverage_config=fixed_coverage_config,
        memory_config=_memory_config(),
    )
    assert report.coverage_report.fully_covered
    assert report.coverage_report.covered_layer_count == _TOTAL_LAYERS
