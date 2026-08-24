"""Unit tests for bench/experiments/mfu_mbu_comparison.py."""

from __future__ import annotations

from modelhub.bench.experiments.mfu_mbu_comparison import (
    ArchitectureMfuMbuComparison,
    MfuMbuComparisonConfig,
    measure_mfu_mbu,
)
from modelhub.monitor.mfu_mbu import BottleneckThresholds
from modelhub.serve.model_profile import ModelProfile

_ARCTIC_PROFILE = ModelProfile.model_validate(
    {
        "model_id": "arctic-text2sql-r1-7b",
        "total_layers": 28,
        "full_attention_layers": 28,
        "gdn_layers": 0,
        "kv_heads": 4,
        "head_dim": 128,
        "kv_dtype_bytes": 2,
        "gdn_fixed_state_bytes": 0,
        "weight_bytes": 16320875725,
    }
)
_QWEN_PROFILE = ModelProfile.model_validate(
    {
        "model_id": "qwen3.5-9b",
        "total_layers": 32,
        "full_attention_layers": 8,
        "gdn_layers": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "kv_dtype_bytes": 2,
        "gdn_fixed_state_bytes": 18874368,
        "weight_bytes": 19327352832,
    }
)


def _config() -> MfuMbuComparisonConfig:
    return MfuMbuComparisonConfig.model_validate(
        {
            "peak_bf16_flops": 209_000_000_000_000.0,
            "peak_bw_bytes_s": 1_792_000_000_000.0,
            "thresholds": {
                "memory_bound_mbu_min": 0.5,
                "memory_bound_mfu_max": 0.3,
                "compute_bound_mfu_min": 0.5,
            },
            "arctic_model_params": 8_160_437_862,
            "qwen_model_params": 9_663_676_416,
        }
    )


def test_measure_mfu_mbu_uses_the_real_a7_formulas() -> None:
    config = _config()
    measurement = measure_mfu_mbu(
        model_id="arctic-text2sql-r1-7b",
        model_params=config.arctic_model_params,
        model_weight_bytes=_ARCTIC_PROFILE.weight_bytes,
        output_tokens_per_s=40.0,
        config=config,
    )
    expected_mfu = (2 * config.arctic_model_params * 40.0) / config.peak_bf16_flops
    expected_mbu = (_ARCTIC_PROFILE.weight_bytes * 40.0) / config.peak_bw_bytes_s
    assert measurement.mfu == expected_mfu
    assert measurement.mbu == expected_mbu


def test_classify_bottleneck_reused_correctly() -> None:
    thresholds = BottleneckThresholds(
        memory_bound_mbu_min=0.5, memory_bound_mfu_max=0.3, compute_bound_mfu_min=0.5
    )
    config = MfuMbuComparisonConfig.model_validate(
        {
            "peak_bf16_flops": 1_000_000.0,
            "peak_bw_bytes_s": 1_000_000.0,
            "thresholds": thresholds.model_dump(),
            "arctic_model_params": 1,
            "qwen_model_params": 1,
        }
    )
    # tiny peak_bf16_flops relative to weight_bytes*tok/s makes this
    # measurement clearly memory-bound (high MBU, low MFU) by construction.
    measurement = measure_mfu_mbu(
        model_id="m",
        model_params=1,
        model_weight_bytes=1_000_000,
        output_tokens_per_s=1.0,
        config=config,
    )
    assert measurement.bottleneck == "memory_bound"


class TestArchitectureMfuMbuComparison:
    def test_kv_bytes_per_token_saved_matches_facts_md(self) -> None:
        config = _config()
        arctic = measure_mfu_mbu(
            model_id="arctic-text2sql-r1-7b",
            model_params=config.arctic_model_params,
            model_weight_bytes=_ARCTIC_PROFILE.weight_bytes,
            output_tokens_per_s=40.0,
            config=config,
        )
        qwen = measure_mfu_mbu(
            model_id="qwen3.5-9b",
            model_params=config.qwen_model_params,
            model_weight_bytes=_QWEN_PROFILE.weight_bytes,
            output_tokens_per_s=40.0,
            config=config,
        )
        comparison = ArchitectureMfuMbuComparison(
            arctic_profile=_ARCTIC_PROFILE,
            qwen_profile=_QWEN_PROFILE,
            arctic_measurement=arctic,
            qwen_measurement=qwen,
        )
        # FACTS.md: Arctic 56 KiB/token, Qwen3.5 32 KiB/token -> 24 KiB saved.
        assert comparison.kv_bytes_per_token_saved_vs_arctic == 24 * 1024

    def test_qwen_mbu_comparison_is_a_real_boolean_not_hardcoded_true(self) -> None:
        config = _config()
        low_mbu_qwen = measure_mfu_mbu(
            model_id="qwen3.5-9b",
            model_params=config.qwen_model_params,
            model_weight_bytes=_QWEN_PROFILE.weight_bytes,
            output_tokens_per_s=0.001,
            config=config,
        )
        high_mbu_arctic = measure_mfu_mbu(
            model_id="arctic-text2sql-r1-7b",
            model_params=config.arctic_model_params,
            model_weight_bytes=_ARCTIC_PROFILE.weight_bytes,
            output_tokens_per_s=1_000_000.0,
            config=config,
        )
        comparison = ArchitectureMfuMbuComparison(
            arctic_profile=_ARCTIC_PROFILE,
            qwen_profile=_QWEN_PROFILE,
            arctic_measurement=high_mbu_arctic,
            qwen_measurement=low_mbu_qwen,
        )
        assert comparison.qwen_mbu_at_least_as_high_as_arctic is False
