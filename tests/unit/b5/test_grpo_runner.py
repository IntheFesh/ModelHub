"""Unit tests for train/grpo/runner.py."""

from __future__ import annotations

import pytest

from modelhub.common.capability import CheckStatus
from modelhub.train.grpo.runner import (
    GrpoTrainingConfig,
    check_verl_available,
    run_grpo_preflight,
    run_grpo_training,
)


def _config(**overrides: object) -> GrpoTrainingConfig:
    defaults: dict[str, object] = {
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "lora_target_modules": ["gate_proj", "q_proj", "in_proj_qkvz"],
        "rollout_k": 8,
        "rollout_max_concurrency": 8,
        "rollout_timeout_s": 30.0,
        "harness_error_abort_threshold": 0.01,
        "seed": 42,
        "max_hours": 8.0,
        "output_dir": "artifacts/train/grpo-qwen3.5-9b",
    }
    defaults.update(overrides)
    return GrpoTrainingConfig.model_validate(defaults)


class TestGrpoTrainingConfig:
    def test_non_positive_rollout_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="rollout_k"):
            _config(rollout_k=0)

    def test_non_positive_max_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_hours"):
            _config(max_hours=0.0)


class TestCheckVerlAvailable:
    def test_reports_a_real_check_status(self) -> None:
        assert check_verl_available() in {CheckStatus.PASS, CheckStatus.SKIP}


class TestRunGrpoPreflight:
    def test_degraded_kernel_status_refuses_to_start(self) -> None:
        # no monkeypatch: this sandbox has no CUDA device, so the real
        # detect_gdn_kernel_status() is honestly degraded=True here.
        with pytest.raises(ValueError, match="refusing to start GRPO rollout"):
            run_grpo_preflight(_config())


class TestRunGrpoTraining:
    def test_raises_when_verl_not_available(self) -> None:
        if check_verl_available() is CheckStatus.PASS:
            pytest.skip("verl is actually installed in this environment")
        with pytest.raises(RuntimeError, match="cannot start GRPO training"):
            run_grpo_training(_config())
