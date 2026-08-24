"""Unit tests for serve/kernel_status.py.

This sandbox genuinely has no GPU and no `torch`/`causal_conv1d`/`fla`
installed (`train` extra not installed — see docs/design-decisions.md) —
so these tests exercise the real SKIP path against real absent
dependencies, not a simulated one. On real GPU hardware with the `train`
extra installed, `check_causal_conv1d`/`check_fla` would instead run the
real CUDA kernel branch (mirroring preflight.py G1/G2) — that branch is
unverified here for the same honest reason `HttpModelClient` is
(CLAUDE.md §12), and is exercised for real once the user runs this on
GPU hardware.
"""

from __future__ import annotations

from modelhub.common.capability import CheckStatus, is_available
from modelhub.serve.kernel_status import (
    check_causal_conv1d,
    check_fla,
    detect_gdn_kernel_status,
    diagnose_gdn_kernels,
)


def test_check_causal_conv1d_is_a_real_skip_in_this_sandbox() -> None:
    result = check_causal_conv1d()
    assert result.status is CheckStatus.SKIP
    assert not is_available(result.status)
    assert result.error is not None


def test_check_fla_is_a_real_skip_in_this_sandbox() -> None:
    result = check_fla()
    assert result.status is CheckStatus.SKIP
    assert not is_available(result.status)
    assert result.error is not None


def test_diagnose_gdn_kernels_reports_both_checks() -> None:
    diagnostics = diagnose_gdn_kernels()
    assert set(diagnostics) == {"causal_conv1d", "fla"}
    for result in diagnostics.values():
        assert result.status is CheckStatus.SKIP


def test_detect_gdn_kernel_status_is_degraded_when_unverified() -> None:
    status = detect_gdn_kernel_status()
    assert status.causal_conv1d is False
    assert status.fla is False
    assert status.degraded is True
