"""Unit tests for bench/cost_model.py."""

from __future__ import annotations

import pytest

from modelhub.bench.cost_model import cost_per_million_tokens_usd, estimate_request_cost_usd


def test_cost_per_million_tokens_basic() -> None:
    # $3.60/hour, 3600 tokens/s -> 3600*3600 = 12,960,000 tokens/hour
    # -> $3.60 / 12,960,000 * 1,000,000 = $0.2778 per 1M tokens
    result = cost_per_million_tokens_usd(gpu_hourly_cost_usd=3.60, tokens_per_s=3600)
    assert result == pytest.approx(0.27777778, rel=1e-4)


def test_higher_throughput_means_lower_cost_per_token() -> None:
    slow = cost_per_million_tokens_usd(gpu_hourly_cost_usd=2.0, tokens_per_s=100)
    fast = cost_per_million_tokens_usd(gpu_hourly_cost_usd=2.0, tokens_per_s=1000)
    assert fast < slow


def test_rejects_non_positive_hourly_cost() -> None:
    with pytest.raises(ValueError, match="gpu_hourly_cost_usd"):
        cost_per_million_tokens_usd(gpu_hourly_cost_usd=0, tokens_per_s=100)


def test_rejects_non_positive_throughput() -> None:
    with pytest.raises(ValueError, match="tokens_per_s"):
        cost_per_million_tokens_usd(gpu_hourly_cost_usd=1.0, tokens_per_s=0)


def test_estimate_request_cost_charges_prompt_and_completion_separately() -> None:
    result = estimate_request_cost_usd(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        gpu_hourly_cost_usd=3.6,
        prompt_tokens_per_s=3600,
        completion_tokens_per_s=900,
    )
    # only prompt tokens charged, at the prompt-tokens-per-s rate
    expected_prompt_only = cost_per_million_tokens_usd(gpu_hourly_cost_usd=3.6, tokens_per_s=3600)
    assert result.cost_usd == pytest.approx(expected_prompt_only)


def test_estimate_request_cost_zero_tokens_is_free() -> None:
    result = estimate_request_cost_usd(
        prompt_tokens=0,
        completion_tokens=0,
        gpu_hourly_cost_usd=3.6,
        prompt_tokens_per_s=3600,
        completion_tokens_per_s=900,
    )
    assert result.cost_usd == 0.0


def test_estimate_request_cost_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_request_cost_usd(
            prompt_tokens=-1,
            completion_tokens=0,
            gpu_hourly_cost_usd=3.6,
            prompt_tokens_per_s=3600,
            completion_tokens_per_s=900,
        )


def test_completion_tokens_typically_cost_more_per_token_than_prompt() -> None:
    # decode throughput is usually much lower than prefill throughput on
    # the same hardware (FACTS.md: decode is memory-bound), so the same
    # dollar-per-hour buys far fewer completion tokens than prompt tokens.
    result = estimate_request_cost_usd(
        prompt_tokens=1000,
        completion_tokens=1000,
        gpu_hourly_cost_usd=3.6,
        prompt_tokens_per_s=3600,
        completion_tokens_per_s=90,
    )
    prompt_only = estimate_request_cost_usd(
        prompt_tokens=1000,
        completion_tokens=0,
        gpu_hourly_cost_usd=3.6,
        prompt_tokens_per_s=3600,
        completion_tokens_per_s=90,
    )
    completion_only_cost = result.cost_usd - prompt_only.cost_usd
    assert completion_only_cost > prompt_only.cost_usd
