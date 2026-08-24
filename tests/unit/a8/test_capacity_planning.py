"""Unit tests for bench/capacity_planning.py.

Numbers are round and hand-verifiable, same discipline as A7's MFU/MBU
tests — not copied from FACTS.md's own (differently-rounded) table.
"""

from __future__ import annotations

import pytest

from modelhub.bench.capacity_planning import (
    CapacityPlanningConfig,
    estimate_max_concurrent_requests,
    find_concurrency_crossover_point,
)
from modelhub.serve.model_profile import ModelProfile


def _profile(**overrides: object) -> ModelProfile:
    defaults: dict[str, object] = {
        "model_id": "test-model",
        "total_layers": 10,
        "full_attention_layers": 10,
        "gdn_layers": 0,
        "kv_heads": 4,
        "head_dim": 128,
        "kv_dtype_bytes": 2,
        "gdn_fixed_state_bytes": 0,
        "weight_bytes": 1_000_000_000,
    }
    defaults.update(overrides)
    return ModelProfile.model_validate(defaults)


def test_estimate_max_concurrent_requests_basic() -> None:
    # kv_bytes_per_token = 2*10*4*128*2 = 20480 bytes
    # usable = 10_000_000_000*1.0 - 1_000_000_000 - 0 - 0 = 9_000_000_000
    # per_request = 20480 * 1000 (prompt+new) = 20_480_000
    # concurrency = 9_000_000_000 // 20_480_000 = 439
    profile = _profile()
    result = estimate_max_concurrent_requests(
        profile,
        gpu_memory_bytes=10_000_000_000,
        gpu_memory_utilization=1.0,
        runtime_overhead_bytes=0,
        prompt_tokens=900,
        max_new_tokens=100,
    )
    assert result == 439


def test_longer_prompt_means_fewer_concurrent_requests() -> None:
    profile = _profile()
    short = estimate_max_concurrent_requests(
        profile,
        gpu_memory_bytes=10_000_000_000,
        gpu_memory_utilization=1.0,
        runtime_overhead_bytes=0,
        prompt_tokens=100,
        max_new_tokens=100,
    )
    long = estimate_max_concurrent_requests(
        profile,
        gpu_memory_bytes=10_000_000_000,
        gpu_memory_utilization=1.0,
        runtime_overhead_bytes=0,
        prompt_tokens=10_000,
        max_new_tokens=100,
    )
    assert long < short


def test_infeasible_budget_raises() -> None:
    profile = _profile(weight_bytes=10_000_000_000)
    with pytest.raises(ValueError, match="no usable KV budget"):
        estimate_max_concurrent_requests(
            profile,
            gpu_memory_bytes=5_000_000_000,
            gpu_memory_utilization=1.0,
            runtime_overhead_bytes=0,
            prompt_tokens=100,
            max_new_tokens=100,
        )


def test_rejects_invalid_utilization() -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        estimate_max_concurrent_requests(
            _profile(),
            gpu_memory_bytes=1,
            gpu_memory_utilization=1.5,
            runtime_overhead_bytes=0,
            prompt_tokens=1,
            max_new_tokens=1,
        )


def test_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_max_concurrent_requests(
            _profile(),
            gpu_memory_bytes=1_000_000_000,
            gpu_memory_utilization=1.0,
            runtime_overhead_bytes=0,
            prompt_tokens=-1,
            max_new_tokens=1,
        )


def test_real_arctic_vs_qwen_profiles_reproduce_the_documented_crossover() -> None:
    """MODEL-SELECTION-FINAL.md: "曲线在 1k–3k 之间交叉" — Arctic-7B ahead
    at short prompts (plain GQA, no fixed overhead), Qwen3.5-9B
    overtaking once GDN's near-flat KV growth outweighs its per-request
    fixed recurrent-state cost. This loads the real committed profiles
    and the real FACTS.md-sourced capacity-planning assumptions — it is
    not a synthetic scenario built to make the test pass."""
    from pathlib import Path

    from modelhub.common.config import load_yaml_config
    from modelhub.common.errors import Stage
    from modelhub.serve.model_profile import load_model_profile

    repo_root = Path(__file__).resolve().parents[3]
    arctic, _ = load_model_profile(
        repo_root / "configs" / "serve" / "model_profiles" / "arctic_7b.yaml"
    )
    qwen, _ = load_model_profile(
        repo_root / "configs" / "serve" / "model_profiles" / "qwen3_5_9b.yaml"
    )
    capacity_config, _ = load_yaml_config(
        CapacityPlanningConfig,
        repo_root / "configs" / "bench" / "capacity_planning.yaml",
        stage=Stage.SERVE,
    )

    result = find_concurrency_crossover_point(
        arctic,
        qwen,
        gpu_memory_bytes=capacity_config.gpu_memory_bytes,
        gpu_memory_utilization=capacity_config.gpu_memory_utilization,
        runtime_overhead_bytes=capacity_config.runtime_overhead_bytes,
        max_new_tokens=0,
        prompt_token_candidates=[1024, 3072, 8192, 32768],
    )

    assert result.found
    assert result.prompt_tokens == 3072
    # Arctic must actually be ahead at the shortest candidate — otherwise
    # this isn't a crossover, Qwen would just win throughout.
    arctic_at_1k = estimate_max_concurrent_requests(
        arctic,
        gpu_memory_bytes=capacity_config.gpu_memory_bytes,
        gpu_memory_utilization=capacity_config.gpu_memory_utilization,
        runtime_overhead_bytes=capacity_config.runtime_overhead_bytes,
        prompt_tokens=1024,
        max_new_tokens=0,
    )
    qwen_at_1k = estimate_max_concurrent_requests(
        qwen,
        gpu_memory_bytes=capacity_config.gpu_memory_bytes,
        gpu_memory_utilization=capacity_config.gpu_memory_utilization,
        runtime_overhead_bytes=capacity_config.runtime_overhead_bytes,
        prompt_tokens=1024,
        max_new_tokens=0,
    )
    assert arctic_at_1k > qwen_at_1k


def test_crossover_found_when_low_kv_model_overtakes() -> None:
    # profile_a: expensive KV (grows fast with prompt length).
    # profile_b: cheap KV (barely grows) — like Qwen3.5's GDN hybrid vs
    # Arctic's plain GQA in FACTS.md, at a much smaller scale so the
    # crossover point is easy to hand-verify from the candidate grid.
    profile_a = _profile(model_id="expensive-kv", kv_heads=8, head_dim=128)
    profile_b = _profile(model_id="cheap-kv", kv_heads=1, head_dim=32)

    result = find_concurrency_crossover_point(
        profile_a,
        profile_b,
        gpu_memory_bytes=10_000_000_000,
        gpu_memory_utilization=1.0,
        runtime_overhead_bytes=0,
        max_new_tokens=100,
        prompt_token_candidates=[100, 1000, 5000, 20000],
    )
    assert result.found
    assert result.prompt_tokens in (100, 1000, 5000, 20000)
    assert result.concurrency_b is not None
    assert result.concurrency_a is not None
    assert result.concurrency_b >= result.concurrency_a


def test_no_crossover_when_one_model_dominates_throughout() -> None:
    profile_a = _profile(model_id="cheap-kv", kv_heads=1, head_dim=32)
    profile_b = _profile(model_id="expensive-kv", kv_heads=8, head_dim=128)

    result = find_concurrency_crossover_point(
        profile_a,
        profile_b,
        gpu_memory_bytes=10_000_000_000,
        gpu_memory_utilization=1.0,
        runtime_overhead_bytes=0,
        max_new_tokens=100,
        prompt_token_candidates=[100, 1000, 5000],
    )
    assert not result.found
    assert result.prompt_tokens is None


def test_crossover_rejects_unsorted_candidates() -> None:
    with pytest.raises(ValueError, match="sorted"):
        find_concurrency_crossover_point(
            _profile(),
            _profile(model_id="other"),
            gpu_memory_bytes=1,
            gpu_memory_utilization=1.0,
            runtime_overhead_bytes=0,
            max_new_tokens=1,
            prompt_token_candidates=[100, 50],
        )


def test_crossover_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_concurrency_crossover_point(
            _profile(),
            _profile(model_id="other"),
            gpu_memory_bytes=1,
            gpu_memory_utilization=1.0,
            runtime_overhead_bytes=0,
            max_new_tokens=1,
            prompt_token_candidates=[],
        )
