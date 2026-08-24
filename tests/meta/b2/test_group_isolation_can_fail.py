"""CLAUDE.md §1.5 required meta-test: proves B2's per-group isolation
(PLAN.md: "每组独立、失败不阻断") actually isolates a real failure
raised by `run_peft_method_comparison_group` (llamafactory-cli not
installed in this sandbox — a real, not simulated, failure) from a
sibling group that completes normally, through the real orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.capability import CheckStatus
from modelhub.train.experiments.orchestrator import run_all_comparison_groups
from modelhub.train.experiments.peft_method_comparison import (
    PeftMethod,
    PeftMethodExperimentConfig,
    check_llamafactory_available,
    run_peft_method_comparison_group,
)


def _peft_config() -> PeftMethodExperimentConfig:
    return PeftMethodExperimentConfig.model_validate(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "dataset_name": "bird23_train_filtered_spider_train",
            "template": "qwen",
            "cutoff_len": 4096,
            "seed": 42,
            "lora_rank": 32,
            "lora_target_modules": ["gate_proj", "q_proj"],
            "qlora_bits": 4,
            "zero3_config_relpath": "artifacts/ds_zero3.json",
            "lora_num_steps": 300,
            "full_num_steps": 200,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "output_root": "artifacts/train/experiments/peft_method",
        }
    )


def test_a_real_llamafactory_unavailable_failure_does_not_block_a_sibling_group(
    tmp_path: Path,
) -> None:
    if check_llamafactory_available().status is CheckStatus.PASS:
        pytest.skip("llamafactory-cli is actually installed in this environment")

    def _failing_lora_group() -> tuple[object, object]:
        metrics = run_peft_method_comparison_group(
            PeftMethod.LORA,
            config=_peft_config(),
            yaml_config_path=tmp_path / "lora.yaml",
            experiment_metrics_path=tmp_path / "experiment_metrics.json",
            gpu_cost_per_hour=2.0,
        )
        return metrics, make_valid_manifest()

    def _ok_sibling_group() -> tuple[object, object]:
        return "sibling completed fine", make_valid_manifest()

    report = run_all_comparison_groups(
        baseline_manifest=make_valid_manifest(),
        groups={"peft_method": _failing_lora_group, "kernel_optimization": _ok_sibling_group},
    )

    assert report.failed_groups == ("peft_method",)
    assert report.completed_groups == ("kernel_optimization",)
    failed = report.outcome_for("peft_method")
    assert failed is not None
    assert "RuntimeError" in (failed.error or "")


def test_all_groups_failing_is_not_the_only_outcome_this_orchestrator_can_produce() -> None:
    """Proves the isolation boundary is not stuck red: an all-healthy
    run of the same orchestrator entry point reports zero failures."""

    def _ok() -> tuple[object, object]:
        return "ok", make_valid_manifest()

    report = run_all_comparison_groups(
        baseline_manifest=make_valid_manifest(),
        groups={"peft_method": _ok, "kernel_optimization": _ok},
    )
    assert report.failed_groups == ()
    assert len(report.completed_groups) == 2
