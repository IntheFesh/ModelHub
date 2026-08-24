"""A8 meta-test: prove `estimate_max_concurrent_requests`'s GDN-cost
semantics are actually right, using the concrete bug this round
shipped-and-fixed as the injected failure (CLAUDE.md §1.5's shape:
inject a scenario a check should classify correctly, assert it does).

While building `tests/unit/a8/test_capacity_planning.py`'s crossover
test, an earlier version of `bench/capacity_planning.py` subtracted
`gdn_fixed_state_bytes` once from the whole GPU's shared usable KV
budget — as if only a single copy of the GDN recurrent state existed
for the entire GPU, rather than one copy per concurrent generation (the
same way KV cache itself is per-sequence). Fed the real committed
Arctic-7B/Qwen3.5-9B profiles, that version put Qwen3.5-9B ahead of
Arctic-7B at every prompt length — flatly contradicting
MODEL-SELECTION-FINAL.md's documented "曲线在 1k–3k 之间交叉" crossover
(Arctic should be ahead at short prompts). This test reimplements that
exact (wrong) formula inline as a local function — never imported by
`src/`, CLAUDE.md §1.4 — and shows it reproduces the contradiction,
then shows the real shipped function does not.
"""

from __future__ import annotations

from pathlib import Path

from modelhub.bench.capacity_planning import (
    CapacityPlanningConfig,
    estimate_max_concurrent_requests,
)
from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.serve.model_profile import ModelProfile, kv_bytes_per_token, load_model_profile

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _buggy_estimate_subtracting_gdn_once(
    profile: ModelProfile,
    *,
    gpu_memory_bytes: int,
    gpu_memory_utilization: float,
    runtime_overhead_bytes: int,
    prompt_tokens: int,
    max_new_tokens: int,
) -> int:
    usable = (
        gpu_memory_bytes * gpu_memory_utilization
        - profile.weight_bytes
        - runtime_overhead_bytes
        - profile.gdn_fixed_state_bytes  # the bug: subtracted once, globally
    )
    per_request = kv_bytes_per_token(profile) * (prompt_tokens + max_new_tokens)
    return int(usable // per_request)


def _load_real_inputs() -> tuple[ModelProfile, ModelProfile, CapacityPlanningConfig]:
    arctic, _ = load_model_profile(
        _REPO_ROOT / "configs" / "serve" / "model_profiles" / "arctic_7b.yaml"
    )
    qwen, _ = load_model_profile(
        _REPO_ROOT / "configs" / "serve" / "model_profiles" / "qwen3_5_9b.yaml"
    )
    capacity, _ = load_yaml_config(
        CapacityPlanningConfig,
        _REPO_ROOT / "configs" / "bench" / "capacity_planning.yaml",
        stage=Stage.SERVE,
    )
    return arctic, qwen, capacity


def test_shipped_formula_matches_crossover_but_the_found_bug_would_not() -> None:
    arctic, qwen, capacity = _load_real_inputs()
    kwargs = {
        "gpu_memory_bytes": capacity.gpu_memory_bytes,
        "gpu_memory_utilization": capacity.gpu_memory_utilization,
        "runtime_overhead_bytes": capacity.runtime_overhead_bytes,
        "prompt_tokens": 1024,
        "max_new_tokens": 0,
    }

    # The actual, shipped formula: Arctic ahead of Qwen at 1k tokens,
    # matching MODEL-SELECTION-FINAL.md's documented crossover direction.
    arctic_real = estimate_max_concurrent_requests(arctic, **kwargs)
    qwen_real = estimate_max_concurrent_requests(qwen, **kwargs)
    assert arctic_real > qwen_real

    # The bug this round found and fixed: subtracting gdn_fixed_state_bytes
    # once globally puts Qwen ahead of Arctic even at 1k tokens, which
    # contradicts the documented crossover entirely.
    arctic_buggy = _buggy_estimate_subtracting_gdn_once(arctic, **kwargs)
    qwen_buggy = _buggy_estimate_subtracting_gdn_once(qwen, **kwargs)
    assert qwen_buggy > arctic_buggy, (
        "test setup assumption broken: the buggy formula was expected to "
        "put Qwen ahead of Arctic even at 1k tokens"
    )


def test_this_meta_test_is_not_always_red_the_two_formulas_can_actually_differ() -> None:
    _arctic, qwen, capacity = _load_real_inputs()
    kwargs = {
        "gpu_memory_bytes": capacity.gpu_memory_bytes,
        "gpu_memory_utilization": capacity.gpu_memory_utilization,
        "runtime_overhead_bytes": capacity.runtime_overhead_bytes,
        "prompt_tokens": 1024,
        "max_new_tokens": 0,
    }
    real = estimate_max_concurrent_requests(qwen, **kwargs)
    buggy = _buggy_estimate_subtracting_gdn_once(qwen, **kwargs)
    assert real != buggy
