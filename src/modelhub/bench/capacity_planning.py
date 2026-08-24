"""KV-budget-based max-concurrency estimation, and the crossover point
between two models' concurrency-vs-prompt-length curves.

Deliberately deferred out of A5 (see docs/design-decisions.md DD-0014):
`serve/model_profile.py` only defines the KV-per-token formula and the
model profiles; the full "how many concurrent requests fit in this GPU's
memory budget" estimation — and finding where two models' curves cross,
which is the routing threshold `gateway/routing.yaml` currently only
holds an ESTIMATE for (DD-0016) — belongs here, in the round PLAN.md
explicitly assigns capacity planning and the crossover point to.

★ `estimate_max_concurrent_requests`'s output is only as good as its
inputs, all of which are estimates pending real measurement
(`serve/model_profile.py`'s `weight_bytes`/`gdn_fixed_state_bytes`,
FACTS.md's `32GB * 0.92 - weights - 2.5GB` runtime-overhead formula).
This module does not silently launder that uncertainty away — it
computes the formula and nothing more, callers/reports must carry the
"estimate, not measured" label forward themselves.

★ `gdn_fixed_state_bytes` is a PER-REQUEST cost, not a one-time
global deduction. An earlier version of this module subtracted it once
from the shared `usable_bytes` pool — that treats the GDN recurrent
state as if only a single copy existed for the whole GPU, but every
concurrent generation needs its own copy of that state (it is per
in-flight sequence, the same way KV cache itself is). That bug had a
concrete, checkable symptom: fed the real committed Arctic-7B/Qwen3.5-9B
profiles, it put Qwen3.5-9B's estimated concurrency ahead of Arctic-7B's
at every prompt length — flatly contradicting MODEL-SELECTION-FINAL.md's
own "曲线在 1k–3k 之间交叉" claim (Arctic ahead at short prompts, Qwen3.5
overtaking past ~1k-3k because GDN's O(1) KV growth eventually wins out
over its fixed per-request state overhead). Charging `gdn_fixed_state_bytes`
per request instead reproduces that crossover almost exactly
(see `tests/unit/a8/test_capacity_planning.py`) — which is itself decent
evidence the fixed-request-cost model is the right one, not just a
retrofit to match a target number.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.common.config import ModelHubBaseConfig
from modelhub.serve.model_profile import ModelProfile, kv_bytes_per_token


class CapacityPlanningConfig(ModelHubBaseConfig):
    """The GPU-memory-budget assumptions FACTS.md's own table uses
    (`configs/bench/capacity_planning.yaml`) — `gpu_memory_utilization`
    and `runtime_overhead_bytes` are FACTS.md's documented `0.92` and
    `2.5GB`, not invented defaults."""

    gpu_memory_bytes: int
    gpu_memory_utilization: float
    runtime_overhead_bytes: int
    max_new_tokens: int


def estimate_max_concurrent_requests(
    profile: ModelProfile,
    *,
    gpu_memory_bytes: int,
    gpu_memory_utilization: float,
    runtime_overhead_bytes: int,
    prompt_tokens: int,
    max_new_tokens: int,
) -> int:
    """usable budget (loaded once, shared by every request) =
    gpu_memory * utilization - weights - runtime_overhead. Each
    concurrent request then costs `kv_bytes_per_token * (prompt_tokens +
    max_new_tokens) + gdn_fixed_state_bytes` — the GDN recurrent state is
    charged per request, not once for the whole GPU (see module
    docstring). concurrency = usable_budget // per_request_bytes."""
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}")
    if prompt_tokens < 0 or max_new_tokens < 0:
        raise ValueError(
            f"prompt_tokens and max_new_tokens must be non-negative, got "
            f"prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens}"
        )

    usable_bytes = (
        gpu_memory_bytes * gpu_memory_utilization - profile.weight_bytes - runtime_overhead_bytes
    )
    if usable_bytes <= 0:
        raise ValueError(
            f"no usable KV budget: weights ({profile.weight_bytes}) + runtime overhead "
            f"({runtime_overhead_bytes}) already exceed {gpu_memory_utilization:.0%} of "
            f"{gpu_memory_bytes} bytes of GPU memory"
        )

    per_request_bytes = (
        kv_bytes_per_token(profile) * (prompt_tokens + max_new_tokens)
        + profile.gdn_fixed_state_bytes
    )
    return int(usable_bytes // per_request_bytes)


@dataclass(frozen=True)
class CrossoverResult:
    found: bool
    prompt_tokens: int | None
    concurrency_a: int | None
    concurrency_b: int | None


def find_concurrency_crossover_point(
    profile_a: ModelProfile,
    profile_b: ModelProfile,
    *,
    gpu_memory_bytes: int,
    gpu_memory_utilization: float,
    runtime_overhead_bytes: int,
    max_new_tokens: int,
    prompt_token_candidates: list[int],
) -> CrossoverResult:
    """Scan `prompt_token_candidates` (ascending) for the first point
    where `profile_b`'s estimated concurrency overtakes `profile_a`'s —
    the KV-economics analogue of FACTS.md's "曲线在 1k–3k 之间交叉"
    finding for Arctic-7B (profile_a) vs. Qwen3.5-9B (profile_b).

    Only ever reports a candidate actually present in
    `prompt_token_candidates` — this is a discrete scan over the caller's
    own grid, not an interpolated estimate the caller didn't ask for.
    `found=False` when no crossover exists in the given range (e.g. one
    model's KV economics dominate throughout), not a fabricated guess.
    """
    if not prompt_token_candidates:
        raise ValueError("prompt_token_candidates must not be empty")
    if sorted(prompt_token_candidates) != prompt_token_candidates:
        raise ValueError("prompt_token_candidates must be sorted ascending")

    def _concurrency(profile: ModelProfile, prompt_tokens: int) -> int:
        return estimate_max_concurrent_requests(
            profile,
            gpu_memory_bytes=gpu_memory_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            runtime_overhead_bytes=runtime_overhead_bytes,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )

    for prompt_tokens in prompt_token_candidates:
        concurrency_a = _concurrency(profile_a, prompt_tokens)
        concurrency_b = _concurrency(profile_b, prompt_tokens)
        if concurrency_b >= concurrency_a:
            return CrossoverResult(True, prompt_tokens, concurrency_a, concurrency_b)

    return CrossoverResult(False, None, None, None)


__all__ = [
    "CapacityPlanningConfig",
    "CrossoverResult",
    "estimate_max_concurrent_requests",
    "find_concurrency_crossover_point",
]
