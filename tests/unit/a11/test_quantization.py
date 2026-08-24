"""Unit tests for bench/experiments/quantization.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import fixed_response_client, make_valid_manifest
from tests.unit.a11.fakes import build_school_db, select_sample

from modelhub.bench.experiments.quantization import (
    QuantizationExperimentConfig,
    QuantizationMethod,
    estimate_concurrency_gain,
    quantize_model_checkpoint,
    quantized_weight_bytes,
    run_quantization_accuracy_check,
    run_truncation_cross_experiment,
)
from modelhub.common.capability import CheckStatus
from modelhub.gate.accuracy_gate import AccuracyGateConfig
from modelhub.gate.admission import AdmissionGateConfig
from modelhub.gate.regression_gate import RegressionGateConfig
from modelhub.gate.safety_gate import SafetyGateConfig
from modelhub.gate.truncation_gate import TruncationGateConfig
from modelhub.gate.types import GateDecision
from modelhub.serve.model_profile import ModelProfile


def _profile(**overrides: object) -> ModelProfile:
    defaults: dict[str, object] = {
        "model_id": "test-model",
        "total_layers": 32,
        "full_attention_layers": 8,
        "gdn_layers": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "kv_dtype_bytes": 2,
        "gdn_fixed_state_bytes": 18874368,
        "weight_bytes": 19327352832,
    }
    defaults.update(overrides)
    return ModelProfile.model_validate(defaults)


def _config(**overrides: object) -> QuantizationExperimentConfig:
    defaults: dict[str, object] = {
        "methods": ["AWQ", "GPTQ", "FP8"],
        "original_bits": 16,
        "gpu_memory_bytes": 34359738368,
        "gpu_memory_utilization": 0.92,
        "runtime_overhead_bytes": 2684354560,
        "prompt_tokens_for_capacity_estimate": 1024,
        "max_new_tokens": 256,
        "generate_timeout_s": 60.0,
        "thinking_modes": [False, True],
    }
    defaults.update(overrides)
    return QuantizationExperimentConfig.model_validate(defaults)


class TestQuantizedWeightBytes:
    def test_awq_4bit_halves_from_8bit_baseline(self) -> None:
        result = quantized_weight_bytes(1000, original_bits=8, target_method=QuantizationMethod.AWQ)
        assert result == 500

    def test_fp8_from_bf16_is_half(self) -> None:
        result = quantized_weight_bytes(
            2_000_000_000, original_bits=16, target_method=QuantizationMethod.FP8
        )
        assert result == 1_000_000_000

    def test_non_positive_original_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            quantized_weight_bytes(0, original_bits=16, target_method=QuantizationMethod.AWQ)

    def test_non_positive_original_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            quantized_weight_bytes(1000, original_bits=0, target_method=QuantizationMethod.AWQ)


class TestEstimateConcurrencyGain:
    def test_quantization_increases_estimated_concurrency(self) -> None:
        result = estimate_concurrency_gain(_profile(), QuantizationMethod.AWQ, _config())
        assert result.concurrency_after > result.concurrency_before
        assert result.concurrency_gain_pct > 0

    def test_smaller_bit_width_method_gains_more_than_fp8(self) -> None:
        awq_result = estimate_concurrency_gain(_profile(), QuantizationMethod.AWQ, _config())
        fp8_result = estimate_concurrency_gain(_profile(), QuantizationMethod.FP8, _config())
        assert awq_result.concurrency_gain_pct > fp8_result.concurrency_gain_pct

    def test_concurrency_gain_pct_zero_before_raises(self) -> None:
        from modelhub.bench.experiments.quantization import QuantizationCapacityResult

        result = QuantizationCapacityResult(
            method=QuantizationMethod.AWQ,
            quantized_weight_bytes=1,
            concurrency_before=0,
            concurrency_after=5,
        )
        with pytest.raises(ValueError, match="cannot express a gain"):
            _ = result.concurrency_gain_pct


class TestQuantizeModelCheckpoint:
    def test_awq_skips_when_autoawq_not_installed(self, tmp_path: Path) -> None:
        result = quantize_model_checkpoint(
            model_path=tmp_path / "in", output_path=tmp_path / "out", method=QuantizationMethod.AWQ
        )
        assert result.status == CheckStatus.SKIP
        assert "autoawq" in result.detail

    def test_fp8_skips_when_llmcompressor_not_installed(self, tmp_path: Path) -> None:
        result = quantize_model_checkpoint(
            model_path=tmp_path / "in", output_path=tmp_path / "out", method=QuantizationMethod.FP8
        )
        assert result.status == CheckStatus.SKIP
        assert "llmcompressor" in result.detail

    def test_gptq_skips_when_gptqmodel_not_installed(self, tmp_path: Path) -> None:
        # In this sandbox the ImportError guard fires before the
        # calibration-dataset NotImplementedError below it is ever
        # reached — that gap is only reachable on a real GPU machine
        # with gptqmodel actually installed, which this test cannot
        # exercise (see the module docstring's honesty note).
        result = quantize_model_checkpoint(
            model_path=tmp_path / "in",
            output_path=tmp_path / "out",
            method=QuantizationMethod.GPTQ,
        )
        assert result.status == CheckStatus.SKIP
        assert "gptqmodel" in result.detail


class TestRunQuantizationAccuracyCheck:
    def test_healthy_quantized_client_passes_admission(self, tmp_path: Path) -> None:
        db_root = build_school_db(tmp_path)
        samples = [select_sample("s0")]
        client = fixed_response_client("SELECT id FROM students")
        admission_config = AdmissionGateConfig(
            accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
            regression=RegressionGateConfig(max_regressions=5),
            safety=SafetyGateConfig(max_allowed_unblocked=0),
            truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
        )
        result = run_quantization_accuracy_check(
            method=QuantizationMethod.AWQ,
            quantized_client=client,
            samples=samples,
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "predictions.jsonl",
            baseline_predictions=None,
            adversarial_samples=[select_sample("adv0", gold_sql="DELETE FROM students")],
            admission_config=admission_config,
            manifest=make_valid_manifest(),
        )
        assert result.method == QuantizationMethod.AWQ
        assert result.metrics.execution_accuracy == 1.0
        assert result.gate_verdict.decision == GateDecision.PASS


class TestRunTruncationCrossExperiment:
    def test_thinking_mode_with_truncated_output_shows_up_in_rate(self, tmp_path: Path) -> None:
        db_root = build_school_db(tmp_path)
        samples = [select_sample(f"s{i}") for i in range(4)]
        cells = [
            (QuantizationMethod.AWQ, False, fixed_response_client("SELECT id FROM students")),
            (
                QuantizationMethod.AWQ,
                True,
                fixed_response_client("SELECT id FROM stud", finish_reason="length"),
            ),
        ]
        results = run_truncation_cross_experiment(
            cells,
            samples=samples,
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_dir=tmp_path,
        )
        no_thinking = next(r for r in results if not r.thinking_enabled)
        thinking = next(r for r in results if r.thinking_enabled)
        assert no_thinking.output_truncated_rate == 0.0
        assert thinking.output_truncated_rate == 1.0
