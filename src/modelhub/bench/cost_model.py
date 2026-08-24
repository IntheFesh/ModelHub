"""GPU-rental-cost -> $/token cost model.

This is the *source* CLAUDE.md §0's "单请求多少钱" number and
`gateway/billing.yaml`'s price table are meant to be derived from: GPU
hourly rental cost divided by real measured throughput
(`bench/load_test.py`'s `LoadTestResult.output_tokens_per_s`), not a
number pulled from a market-rate lookup — `gateway/billing.py` (A6) is
the *consumer* of a price table, this module is what actually computes
that table from hardware economics.

★ No default GPU hourly rate is baked in anywhere in this module.
FACTS.md only documents a rental-cost range for the *training* GPUs
(2xA100-80G, "$112–180 overseas / ¥450–630 domestic" for the whole SFT
run) — there is no equivalent confirmed figure for the RTX 5090 serving
GPU this cost model is actually for. Every function here takes
`gpu_hourly_cost_usd` as a required argument; inventing a plausible-
looking default would be exactly the kind of unverified number CLAUDE.md
Section 12 forbids treating as settled.
"""

from __future__ import annotations

from dataclasses import dataclass


def cost_per_million_tokens_usd(*, gpu_hourly_cost_usd: float, tokens_per_s: float) -> float:
    if gpu_hourly_cost_usd <= 0:
        raise ValueError(f"gpu_hourly_cost_usd must be positive, got {gpu_hourly_cost_usd}")
    if tokens_per_s <= 0:
        raise ValueError(f"tokens_per_s must be positive, got {tokens_per_s}")
    tokens_per_hour = tokens_per_s * 3600
    return (gpu_hourly_cost_usd / tokens_per_hour) * 1_000_000


@dataclass(frozen=True)
class RequestCostEstimate:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def estimate_request_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    gpu_hourly_cost_usd: float,
    prompt_tokens_per_s: float,
    completion_tokens_per_s: float,
) -> RequestCostEstimate:
    """Prompt and completion tokens are costed at their own measured
    throughput (prefill and decode have very different tok/s on the same
    hardware — FACTS.md's MFU/MBU discussion is exactly why), not a
    single blended rate."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError(
            f"token counts must be non-negative, got prompt={prompt_tokens} "
            f"completion={completion_tokens}"
        )
    prompt_rate = cost_per_million_tokens_usd(
        gpu_hourly_cost_usd=gpu_hourly_cost_usd, tokens_per_s=prompt_tokens_per_s
    )
    completion_rate = cost_per_million_tokens_usd(
        gpu_hourly_cost_usd=gpu_hourly_cost_usd, tokens_per_s=completion_tokens_per_s
    )
    cost_usd = (prompt_tokens / 1_000_000) * prompt_rate + (
        completion_tokens / 1_000_000
    ) * completion_rate
    return RequestCostEstimate(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost_usd
    )


__all__ = ["RequestCostEstimate", "cost_per_million_tokens_usd", "estimate_request_cost_usd"]
