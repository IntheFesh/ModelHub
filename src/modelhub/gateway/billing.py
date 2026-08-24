"""Per-request cost accounting.

CLAUDE.md §0 names "单请求多少钱" (cost per request) as one of the
judged, must-be-traceable numbers this whole project exists to produce.
This module computes that number from real token counts and a
config-driven price table (`configs/gateway/billing.yaml`) — it does not
estimate GPU-hour amortized cost or anything requiring a live serving run
(that belongs to A8's cost model). This is deliberately the narrower,
per-request "prompt tokens × price + completion tokens × price" number.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.common.config import ModelHubBaseConfig


class BillingConfig(ModelHubBaseConfig):
    # USD per 1,000 tokens, keyed by model_id. Separate prompt/completion
    # rates because they're priced differently everywhere in this
    # industry (completion tokens cost more compute per token to
    # generate than prompt tokens cost to encode).
    price_per_1k_prompt_tokens_usd: dict[str, float]
    price_per_1k_completion_tokens_usd: dict[str, float]


@dataclass(frozen=True)
class RequestCost:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.prompt_cost_usd + self.completion_cost_usd


def compute_request_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int, config: BillingConfig
) -> RequestCost:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError(
            f"token counts must be non-negative, got prompt={prompt_tokens} "
            f"completion={completion_tokens}"
        )
    if model_id not in config.price_per_1k_prompt_tokens_usd:
        raise ValueError(f"no configured prompt-token price for model_id {model_id!r}")
    if model_id not in config.price_per_1k_completion_tokens_usd:
        raise ValueError(f"no configured completion-token price for model_id {model_id!r}")

    prompt_rate = config.price_per_1k_prompt_tokens_usd[model_id]
    completion_rate = config.price_per_1k_completion_tokens_usd[model_id]
    return RequestCost(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cost_usd=(prompt_tokens / 1000) * prompt_rate,
        completion_cost_usd=(completion_tokens / 1000) * completion_rate,
    )


__all__ = ["BillingConfig", "RequestCost", "compute_request_cost"]
