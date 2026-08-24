"""MFU/MBU: the two hardware-utilization numbers this project actually
reports, instead of `DCGM_FI_DEV_GPU_UTIL` / nvidia-smi's
`utilization.gpu`.

FACTS.md §四: that field's official definition is "the fraction of the
sampling period during which at least one kernel was executing" — a
time-occupancy flag, not a workload measure. A single small kernel
occupying one SM for the entire sampling window reads 100% while every
other SM sits idle, which is the *normal* case for LLM inference, not an
anomaly. Reporting it as "GPU utilization: 95%" invites (and deserves)
the question "which utilization do you mean."

    MFU = (2 * params * output_tok_s) / peak_bf16_flops
    MBU = (weight_bytes * output_tok_s) / peak_bw_bytes_s

Both peak values MUST come from a real measurement
(`hardware_bench.py`'s GEMM/memcpy microbenchmark, or a manifest's
`measured_peak_tflops`/`measured_bw_gbs`) — there is deliberately no
"use the spec-sheet number if unmeasured" fallback baked into these
functions; a caller with only a spec value must pass it in knowing it
isn't measured, not get it silently substituted.

FACTS.md's own diagnostic reading: high MBU + low MFU during decode is
normal and expected (LLM decode is memory-bound, not compute-bound);
high MFU during prefill indicates the opposite. `classify_bottleneck`
turns that qualitative reading into an explicit, config-driven decision
rather than a threshold buried in a report-writing function somewhere.
"""

from __future__ import annotations

from typing import Literal

from modelhub.common.config import ModelHubBaseConfig

BottleneckClass = Literal["memory_bound", "compute_bound", "balanced"]


class BottleneckThresholds(ModelHubBaseConfig):
    memory_bound_mbu_min: float
    memory_bound_mfu_max: float
    compute_bound_mfu_min: float


def compute_mfu(*, model_params: int, output_tokens_per_s: float, peak_bf16_flops: float) -> float:
    """Model FLOPs Utilization: fraction of peak compute actually used."""
    if model_params <= 0:
        raise ValueError(f"model_params must be positive, got {model_params}")
    if output_tokens_per_s < 0:
        raise ValueError(f"output_tokens_per_s must be non-negative, got {output_tokens_per_s}")
    if peak_bf16_flops <= 0:
        raise ValueError(f"peak_bf16_flops must be positive, got {peak_bf16_flops}")
    return (2 * model_params * output_tokens_per_s) / peak_bf16_flops


def compute_mbu(
    *, model_weight_bytes: int, output_tokens_per_s: float, peak_bw_bytes_s: float
) -> float:
    """Model Bandwidth Utilization: fraction of peak memory bandwidth
    actually used re-reading model weights once per generated token."""
    if model_weight_bytes <= 0:
        raise ValueError(f"model_weight_bytes must be positive, got {model_weight_bytes}")
    if output_tokens_per_s < 0:
        raise ValueError(f"output_tokens_per_s must be non-negative, got {output_tokens_per_s}")
    if peak_bw_bytes_s <= 0:
        raise ValueError(f"peak_bw_bytes_s must be positive, got {peak_bw_bytes_s}")
    return (model_weight_bytes * output_tokens_per_s) / peak_bw_bytes_s


def classify_bottleneck(
    mfu: float, mbu: float, thresholds: BottleneckThresholds
) -> BottleneckClass:
    """`memory_bound` is checked first: FACTS.md's normal-decode reading
    is "MBU high AND MFU low," and that combination should win over a
    borderline compute_bound reading on the same sample."""
    if mbu >= thresholds.memory_bound_mbu_min and mfu <= thresholds.memory_bound_mfu_max:
        return "memory_bound"
    if mfu >= thresholds.compute_bound_mfu_min:
        return "compute_bound"
    return "balanced"


__all__ = [
    "BottleneckClass",
    "BottleneckThresholds",
    "classify_bottleneck",
    "compute_mbu",
    "compute_mfu",
]
