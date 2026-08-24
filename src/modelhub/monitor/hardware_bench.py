"""Real measured hardware peak BF16 TFLOPS / memory bandwidth.

Refactored from `preflight.py`'s E1/E2 microbenchmarks (a real
BF16 GEMM and a real device-to-device memcpy, timed) into importable
functions — the same relationship A5's `serve/kernel_status.py` has to
`preflight.py`'s G1/G2: one real implementation, reusable by the
standalone preflight CLI and by anything that wants a manifest's
`measured_peak_tflops`/`measured_bw_gbs` populated with an actual number
instead of the un-calibrated 209 TFLOPS / 1792 GB/s spec-sheet assumption
(CLAUDE.md §3.1: `measured_peak_tflops`/`measured_bw_gbs` are explicit
manifest fields precisely so a report can tell "measured" from "assumed"
apart).

This sandbox has no GPU (no `torch`, confirmed absent — see
docs/design-decisions.md), so both functions here exercise the real SKIP
path in every test that runs against them, not a simulated one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from modelhub.common.capability import CheckStatus

_TERA = 1e12
_GIGA = 1e9


@dataclass(frozen=True)
class TflopsMeasurement:
    status: CheckStatus
    tflops: float | None
    detail: str


@dataclass(frozen=True)
class BandwidthMeasurement:
    status: CheckStatus
    gbs: float | None
    detail: str


def measure_bf16_tflops(*, matrix_size: int = 8192, iterations: int = 20) -> TflopsMeasurement:
    """A square BF16 GEMM (`matrix_size x matrix_size`), timed after a
    warmup, converted to TFLOPS via `2 * n^3` FLOPs per matmul."""
    try:
        import torch
    except ImportError as e:
        return TflopsMeasurement(
            CheckStatus.SKIP, None, f"torch not installed: {type(e).__name__}: {e}"
        )
    if not torch.cuda.is_available():
        return TflopsMeasurement(CheckStatus.SKIP, None, "no CUDA device available")

    try:
        n = matrix_size
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            _ = a @ a
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(iterations):
            _ = a @ a
        torch.cuda.synchronize()
        elapsed_s = time.monotonic() - t0
    except Exception as e:
        return TflopsMeasurement(
            CheckStatus.FAIL, None, f"benchmark raised: {type(e).__name__}: {e}"
        )

    if elapsed_s <= 0:
        return TflopsMeasurement(CheckStatus.FAIL, None, "measured elapsed time was non-positive")
    tflops = (2 * n**3 * iterations) / elapsed_s / _TERA
    return TflopsMeasurement(CheckStatus.PASS, tflops, f"measured {tflops:.1f} TFLOPS (BF16 GEMM)")


def measure_memory_bandwidth_gbs(
    *, elements: int = 256 * 1024 * 1024, iterations: int = 20
) -> BandwidthMeasurement:
    """A device-to-device `float16` copy, timed after a warmup. `2x`
    element bytes per copy (one read, one write)."""
    try:
        import torch
    except ImportError as e:
        return BandwidthMeasurement(
            CheckStatus.SKIP, None, f"torch not installed: {type(e).__name__}: {e}"
        )
    if not torch.cuda.is_available():
        return BandwidthMeasurement(CheckStatus.SKIP, None, "no CUDA device available")

    try:
        x = torch.empty(elements, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(iterations):
            y.copy_(x)
        torch.cuda.synchronize()
        elapsed_s = time.monotonic() - t0
    except Exception as e:
        return BandwidthMeasurement(
            CheckStatus.FAIL, None, f"benchmark raised: {type(e).__name__}: {e}"
        )

    if elapsed_s <= 0:
        return BandwidthMeasurement(
            CheckStatus.FAIL, None, "measured elapsed time was non-positive"
        )
    gbs = (2 * x.numel() * x.element_size() * iterations) / elapsed_s / _GIGA
    return BandwidthMeasurement(CheckStatus.PASS, gbs, f"measured {gbs:.0f} GB/s (D2D copy)")


__all__ = [
    "BandwidthMeasurement",
    "TflopsMeasurement",
    "measure_bf16_tflops",
    "measure_memory_bandwidth_gbs",
]
