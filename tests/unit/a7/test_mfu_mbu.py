"""Unit tests for monitor/mfu_mbu.py.

Every expected value below is chosen so it can be verified by hand from
the formula in the module docstring, not copied from the implementation.
"""

from __future__ import annotations

import pytest

from modelhub.monitor.mfu_mbu import (
    BottleneckThresholds,
    classify_bottleneck,
    compute_mbu,
    compute_mfu,
)


def test_compute_mfu_full_utilization() -> None:
    # MFU = 2 * 1e9 * 50 / 1e11 = 1e11 / 1e11 = 1.0
    result = compute_mfu(model_params=1_000_000_000, output_tokens_per_s=50, peak_bf16_flops=1e11)
    assert result == pytest.approx(1.0)


def test_compute_mfu_half_utilization() -> None:
    # MFU = 2 * 1e9 * 25 / 1e11 = 5e10 / 1e11 = 0.5
    result = compute_mfu(model_params=1_000_000_000, output_tokens_per_s=25, peak_bf16_flops=1e11)
    assert result == pytest.approx(0.5)


def test_compute_mfu_zero_throughput_is_zero() -> None:
    result = compute_mfu(model_params=1_000_000_000, output_tokens_per_s=0, peak_bf16_flops=1e11)
    assert result == 0.0


def test_compute_mfu_rejects_non_positive_params() -> None:
    with pytest.raises(ValueError, match="model_params"):
        compute_mfu(model_params=0, output_tokens_per_s=1, peak_bf16_flops=1e11)


def test_compute_mfu_rejects_negative_throughput() -> None:
    with pytest.raises(ValueError, match="output_tokens_per_s"):
        compute_mfu(model_params=1, output_tokens_per_s=-1, peak_bf16_flops=1e11)


def test_compute_mfu_rejects_non_positive_peak_flops() -> None:
    with pytest.raises(ValueError, match="peak_bf16_flops"):
        compute_mfu(model_params=1, output_tokens_per_s=1, peak_bf16_flops=0)


def test_compute_mbu_full_utilization() -> None:
    # MBU = 2e9 * 50 / 1e11 = 1e11 / 1e11 = 1.0
    result = compute_mbu(
        model_weight_bytes=2_000_000_000, output_tokens_per_s=50, peak_bw_bytes_s=1e11
    )
    assert result == pytest.approx(1.0)


def test_compute_mbu_quarter_utilization() -> None:
    # MBU = 2e9 * 12.5 / 1e11 = 2.5e10 / 1e11 = 0.25
    result = compute_mbu(
        model_weight_bytes=2_000_000_000, output_tokens_per_s=12.5, peak_bw_bytes_s=1e11
    )
    assert result == pytest.approx(0.25)


def test_compute_mbu_rejects_non_positive_weight_bytes() -> None:
    with pytest.raises(ValueError, match="model_weight_bytes"):
        compute_mbu(model_weight_bytes=0, output_tokens_per_s=1, peak_bw_bytes_s=1e11)


def test_compute_mbu_rejects_non_positive_peak_bw() -> None:
    with pytest.raises(ValueError, match="peak_bw_bytes_s"):
        compute_mbu(model_weight_bytes=1, output_tokens_per_s=1, peak_bw_bytes_s=0)


def _thresholds() -> BottleneckThresholds:
    return BottleneckThresholds(
        memory_bound_mbu_min=0.5, memory_bound_mfu_max=0.3, compute_bound_mfu_min=0.5
    )


def test_classify_decode_like_profile_is_memory_bound() -> None:
    # FACTS.md: "MBU 高 + MFU 低 = memory-bound" is the normal decode reading.
    assert classify_bottleneck(mfu=0.1, mbu=0.8, thresholds=_thresholds()) == "memory_bound"


def test_classify_prefill_like_profile_is_compute_bound() -> None:
    assert classify_bottleneck(mfu=0.8, mbu=0.2, thresholds=_thresholds()) == "compute_bound"


def test_classify_ambiguous_profile_is_balanced() -> None:
    assert classify_bottleneck(mfu=0.4, mbu=0.4, thresholds=_thresholds()) == "balanced"


def test_classify_boundary_values_are_inclusive() -> None:
    assert classify_bottleneck(mfu=0.3, mbu=0.5, thresholds=_thresholds()) == "memory_bound"


def test_classify_high_mfu_and_high_mbu_favors_compute_bound() -> None:
    # memory_bound requires mfu <= 0.3, which fails here even though mbu
    # is also high — the classifier isn't just "whichever is bigger."
    assert classify_bottleneck(mfu=0.9, mbu=0.9, thresholds=_thresholds()) == "compute_bound"
