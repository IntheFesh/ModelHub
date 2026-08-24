"""Group 2: prefix caching — on/off hit rate, request-interleaving
degree's effect on hit rate, and the v2-added Arctic (standard KV prefix
reuse) vs Qwen3.5 (only 8 of 32 layers hold reusable prefix KV; the other
24 GDN layers are recurrent state) comparison.

★ PLAN.md's core warning: "不要用同一个db连发1000次刷99%命中率" — real
traffic interleaves requests across different db schemas, and different
schemas evict each other's cached prefix under any bounded-size cache.
`build_interleaved_traffic` exists specifically so this experiment
measures that realistic case, not the easy one.

★ Hit rate must come from vLLM's own Prometheus counters
(`vllm_metrics.py`), never estimated client-side.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

from modelhub.bench.experiments.vllm_metrics import fetch_counter
from modelhub.bench.load_test import LoadTestResult
from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.model_client import ModelClient
from modelhub.monitor.cache_metrics import PrefixCacheStats


class PrefixCacheExperimentConfig(ModelHubBaseConfig):
    metrics_url: str
    hits_metric_name: str
    queries_metric_name: str
    num_distinct_dbs: int
    requests_per_db: int
    interleaving_degrees: list[int]
    max_tokens: int
    temperature: float
    generate_timeout_s: float


@dataclass(frozen=True)
class InterleavedRequest:
    db_id: str
    prompt: str


def build_interleaved_traffic(
    prompts_by_db: dict[str, list[str]], *, interleaving_degree: int
) -> list[InterleavedRequest]:
    """Round-robin across every `db_id` in `prompts_by_db`, sending
    `interleaving_degree` consecutive requests to the same db before
    switching to the next. `interleaving_degree=1` is the worst case for
    a plain LRU prefix cache (every request switches schema); a value at
    least as large as the longest per-db queue reproduces PLAN.md's
    explicitly-forbidden easy case ("同一个db连发N次")."""
    if interleaving_degree < 1:
        raise ValueError(f"interleaving_degree must be >= 1, got {interleaving_degree}")
    if not prompts_by_db:
        raise ValueError("prompts_by_db must not be empty")

    queues = {db_id: list(prompts) for db_id, prompts in prompts_by_db.items()}
    sequence: list[InterleavedRequest] = []
    remaining = sum(len(q) for q in queues.values())
    db_cycle = cycle(sorted(queues))
    while remaining > 0:
        db_id = next(db_cycle)
        queue = queues[db_id]
        for _ in range(interleaving_degree):
            if not queue:
                break
            sequence.append(InterleavedRequest(db_id=db_id, prompt=queue.pop(0)))
            remaining -= 1
    return sequence


def run_prefix_cache_traffic(
    model_client: ModelClient,
    traffic: list[InterleavedRequest],
    *,
    max_tokens: int,
    temperature: float,
    generate_timeout_s: float,
) -> None:
    """Fire every request in `traffic` sequentially, in order. Cache
    behavior is an artifact of request ORDER — a concurrent pool
    (`bench/load_test.py`'s harness) could reorder or interleave requests
    across threads and invalidate the interleaving-degree the caller just
    constructed, so this deliberately runs single-threaded."""
    for request in traffic:
        model_client.generate(
            request.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=generate_timeout_s,
        )


def measure_prefix_cache_hit_rate(
    *, metrics_url: str, hits_metric_name: str, queries_metric_name: str
) -> PrefixCacheStats:
    """Real vLLM Prometheus counters only (CLAUDE.md: "命中率必须来自
    vLLM 真实 metrics") — raises rather than substituting 0 if either
    counter is genuinely absent from the scrape."""
    hits = fetch_counter(metrics_url, hits_metric_name)
    queries = fetch_counter(metrics_url, queries_metric_name)
    if hits is None or queries is None:
        raise ValueError(
            f"vLLM did not report prefix-cache counters {hits_metric_name!r}/"
            f"{queries_metric_name!r} at {metrics_url} — refusing to compute a hit "
            f"rate from a missing metric (CLAUDE.md §1.1: missing is not 0)"
        )
    return PrefixCacheStats(hits=int(hits), total_requests=int(queries))


@dataclass(frozen=True)
class PrefixCacheOnOffResult:
    interleaving_degree: int
    cache_enabled_stats: PrefixCacheStats
    cache_enabled_load: LoadTestResult
    cache_disabled_load: LoadTestResult

    @property
    def latency_p50_savings_s(self) -> float | None:
        enabled = self.cache_enabled_load.latency_p50_s
        disabled = self.cache_disabled_load.latency_p50_s
        if enabled is None or disabled is None:
            return None
        return disabled - enabled


@dataclass(frozen=True)
class ArchitectureCacheComparison:
    """v2-added: Arctic (dense attention, every layer's KV is reusable)
    vs Qwen3.5 (only 8/32 layers are full-attention; the 24 GDN layers
    carry recurrent state with nothing to prefix-cache). This dataclass
    only carries the two measured `PrefixCacheStats` side by side — it
    computes nothing that isn't a direct function of real measurements."""

    arctic_stats: PrefixCacheStats
    qwen_stats: PrefixCacheStats

    @property
    def hit_rate_discount(self) -> float | None:
        """Arctic hit rate minus Qwen3.5 hit rate under identical
        traffic — positive means the hybrid architecture's cache benefit
        is discounted, as MODEL-SELECTION-FINAL.md predicts. `None` when
        either side has no measured hit rate."""
        if self.arctic_stats.hit_rate is None or self.qwen_stats.hit_rate is None:
            return None
        return self.arctic_stats.hit_rate - self.qwen_stats.hit_rate


__all__ = [
    "ArchitectureCacheComparison",
    "InterleavedRequest",
    "PrefixCacheExperimentConfig",
    "PrefixCacheOnOffResult",
    "build_interleaved_traffic",
    "measure_prefix_cache_hit_rate",
    "run_prefix_cache_traffic",
]
