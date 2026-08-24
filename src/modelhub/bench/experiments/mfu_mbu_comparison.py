"""Group 6 (v2-added): MFU/MBU architecture comparison — Arctic-7B
(dense attention) vs Qwen3.5-9B (GDN hybrid) under the same serving
conditions. Reuses `monitor/mfu_mbu.py`'s `compute_mfu`/`compute_mbu`/
`classify_bottleneck` (A7) unchanged; this module only pairs two models'
measurements side by side.

★ The claim this experiment exists to test, in PLAN.md's own words:
"Qwen3.5 省的是显存容量，不是显存带宽" — two DIFFERENT resources, easy to
conflate:
  - VRAM CAPACITY: `serve/model_profile.py::kv_bytes_per_token` (A5) —
    Qwen3.5's 24 GDN layers hold an O(1) recurrent state instead of a
    per-token KV entry, so its KV/token (32 KiB) is lower than Arctic's
    (56 KiB). This is the side Qwen3.5 actually wins on — more requests'
    worth of KV cache fit in the same GPU (A8's capacity planning).
  - VRAM BANDWIDTH: MBU, measured here. GDN decode's arithmetic
    intensity is lower than standard attention's (FACTS.md), so there is
    no reason to expect Qwen3.5 to look LESS memory-bandwidth-bound
    during decode — if anything, comparably or more so. Conflating the
    two into one "Qwen3.5 saves memory" claim would overstate what the
    KV-capacity win actually buys.
`ArchitectureMfuMbuComparison` reports both numbers explicitly rather
than a single combined score, so the capacity-vs-bandwidth distinction
stays visible in the result, not just in this docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.common.config import ModelHubBaseConfig
from modelhub.monitor.mfu_mbu import (
    BottleneckClass,
    BottleneckThresholds,
    classify_bottleneck,
    compute_mbu,
    compute_mfu,
)
from modelhub.serve.model_profile import ModelProfile, kv_bytes_per_token


class MfuMbuComparisonConfig(ModelHubBaseConfig):
    # measured values (hardware_bench.py's real GEMM/memcpy microbenchmark
    # or a manifest's measured_peak_tflops/measured_bw_gbs) — no
    # spec-sheet fallback, matching compute_mfu/compute_mbu's own
    # no-silent-default contract.
    peak_bf16_flops: float
    peak_bw_bytes_s: float
    thresholds: BottleneckThresholds
    # param counts derived from weight_bytes assuming bf16 storage
    # (weight_bytes // 2) — an ESTIMATE pending the real count from each
    # model's config.json, not a measured value.
    arctic_model_params: int
    qwen_model_params: int


@dataclass(frozen=True)
class ModelMfuMbuMeasurement:
    model_id: str
    model_params: int
    output_tokens_per_s: float
    mfu: float
    mbu: float
    bottleneck: BottleneckClass


def measure_mfu_mbu(
    *,
    model_id: str,
    model_params: int,
    model_weight_bytes: int,
    output_tokens_per_s: float,
    config: MfuMbuComparisonConfig,
) -> ModelMfuMbuMeasurement:
    mfu = compute_mfu(
        model_params=model_params,
        output_tokens_per_s=output_tokens_per_s,
        peak_bf16_flops=config.peak_bf16_flops,
    )
    mbu = compute_mbu(
        model_weight_bytes=model_weight_bytes,
        output_tokens_per_s=output_tokens_per_s,
        peak_bw_bytes_s=config.peak_bw_bytes_s,
    )
    return ModelMfuMbuMeasurement(
        model_id=model_id,
        model_params=model_params,
        output_tokens_per_s=output_tokens_per_s,
        mfu=mfu,
        mbu=mbu,
        bottleneck=classify_bottleneck(mfu, mbu, config.thresholds),
    )


@dataclass(frozen=True)
class ArchitectureMfuMbuComparison:
    arctic_profile: ModelProfile
    qwen_profile: ModelProfile
    arctic_measurement: ModelMfuMbuMeasurement
    qwen_measurement: ModelMfuMbuMeasurement

    @property
    def kv_bytes_per_token_saved_vs_arctic(self) -> int:
        """The VRAM-CAPACITY half of the claim — positive means Qwen3.5
        needs fewer KV-cache bytes per token than Arctic (A5/FACTS.md's
        documented 56 KiB -> 32 KiB)."""
        return kv_bytes_per_token(self.arctic_profile) - kv_bytes_per_token(self.qwen_profile)

    @property
    def qwen_mbu_at_least_as_high_as_arctic(self) -> bool:
        """The VRAM-BANDWIDTH half of the claim, made numeric: True means
        Qwen3.5 is NOT getting a bandwidth break the way it gets a
        capacity break — GDN's lower arithmetic intensity should keep it
        comparably or more memory-bound, not less. This property states
        the comparison; it does not assert the expected direction is
        real — a run that measures the opposite disproves PLAN.md's
        claim, and should be reported as such, not discarded."""
        return self.qwen_measurement.mbu >= self.arctic_measurement.mbu


__all__ = [
    "ArchitectureMfuMbuComparison",
    "MfuMbuComparisonConfig",
    "ModelMfuMbuMeasurement",
    "measure_mfu_mbu",
]
