"""Unit tests for monitor/hardware_bench.py.

This sandbox has no GPU/torch (see docs/design-decisions.md) — these
tests exercise the real SKIP path, same as tests/unit/a5/test_kernel_status.py.
"""

from __future__ import annotations

from modelhub.common.capability import CheckStatus, is_available
from modelhub.monitor.hardware_bench import measure_bf16_tflops, measure_memory_bandwidth_gbs


def test_measure_bf16_tflops_is_a_real_skip_in_this_sandbox() -> None:
    result = measure_bf16_tflops()
    assert result.status is CheckStatus.SKIP
    assert not is_available(result.status)
    assert result.tflops is None


def test_measure_memory_bandwidth_is_a_real_skip_in_this_sandbox() -> None:
    result = measure_memory_bandwidth_gbs()
    assert result.status is CheckStatus.SKIP
    assert not is_available(result.status)
    assert result.gbs is None


def test_measurements_never_fabricate_a_number_when_skipped() -> None:
    tflops_result = measure_bf16_tflops()
    bw_result = measure_memory_bandwidth_gbs()
    # CLAUDE.md Section 1.1: missing is None, never a fake 0.0.
    assert tflops_result.tflops is None
    assert bw_result.gbs is None
