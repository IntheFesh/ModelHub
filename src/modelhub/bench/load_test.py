"""Concurrent load-testing harness: fires `concurrency` workers against a
`ModelClient`, each issuing requests until `num_requests` total
completions (successes + failures) are reached, and reports QPS +
latency percentiles.

Uses the same `ModelClient` Protocol `eval/model_client.py` defines —
this harness is backend-agnostic (a real `HttpModelClient` against a
served vLLM instance, or the fake client tests use) rather than
reimplementing its own narrower request interface. A0/A4's precedent
carries over directly: mocks live only in `tests/`, this module accepts
any real `ModelClient` implementation.

Percentiles use the nearest-rank method over the full sorted latency
sample — no numpy dependency for something this simple, and no silent
truncation of the sample. A failed request is counted and classified by
exception type (`LoadTestResult.error_breakdown`), never silently
dropped from the denominator — the same "missing ≠ 0, failure ≠ absence"
discipline `eval/` applies to SQL execution outcomes.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.model_client import ModelClient


class LoadTestConfig(ModelHubBaseConfig):
    concurrency: int
    num_requests: int
    max_tokens: int
    temperature: float
    generate_timeout_s: float
    # the sweep bucket this run belongs to (e.g. "1k", "3k", "8k") — a
    # label, not a token count this harness measures itself; the caller
    # is responsible for constructing a `prompt` of roughly that length.
    prompt_length_label: str


@dataclass(frozen=True)
class LoadTestResult:
    concurrency: int
    num_requests: int
    prompt_length_label: str
    succeeded: int
    failed: int
    error_breakdown: dict[str, int]
    total_wall_time_s: float
    qps: float
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    output_tokens_per_s: float | None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile; `pct` in `[0, 100]`, `sorted_values` must
    be non-empty and already sorted ascending."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def run_load_test(model_client: ModelClient, prompt: str, config: LoadTestConfig) -> LoadTestResult:
    latencies: list[float] = []
    error_breakdown: dict[str, int] = {}
    prompt_tokens_total = 0
    completion_tokens_total = 0
    lock = threading.Lock()

    def _one_request(_index: int) -> None:
        nonlocal prompt_tokens_total, completion_tokens_total
        t0 = time.monotonic()
        try:
            result = model_client.generate(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout_s=config.generate_timeout_s,
            )
        except Exception as e:
            # A load-testing harness must not crash on the first backend
            # error mid-sweep — every failure is counted and classified
            # by exception type rather than swallowed (CLAUDE.md Section
            # 1.1), and re-raises are never appropriate here since the
            # whole point is measuring the failure rate under load.
            with lock:
                error_breakdown[type(e).__name__] = error_breakdown.get(type(e).__name__, 0) + 1
            return
        elapsed_s = time.monotonic() - t0
        with lock:
            latencies.append(elapsed_s)
            prompt_tokens_total += result.prompt_tokens or 0
            completion_tokens_total += result.completion_tokens or 0

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(_one_request, range(config.num_requests)))
    total_wall_time_s = time.monotonic() - t_start

    succeeded = len(latencies)
    failed = config.num_requests - succeeded
    sorted_latencies = sorted(latencies)
    qps = succeeded / total_wall_time_s if total_wall_time_s > 0 else 0.0
    output_tokens_per_s = (
        completion_tokens_total / total_wall_time_s
        if total_wall_time_s > 0 and completion_tokens_total > 0
        else None
    )

    return LoadTestResult(
        concurrency=config.concurrency,
        num_requests=config.num_requests,
        prompt_length_label=config.prompt_length_label,
        succeeded=succeeded,
        failed=failed,
        error_breakdown=error_breakdown,
        total_wall_time_s=total_wall_time_s,
        qps=qps,
        latency_p50_s=_percentile(sorted_latencies, 50) if sorted_latencies else None,
        latency_p95_s=_percentile(sorted_latencies, 95) if sorted_latencies else None,
        latency_p99_s=_percentile(sorted_latencies, 99) if sorted_latencies else None,
        total_prompt_tokens=prompt_tokens_total,
        total_completion_tokens=completion_tokens_total,
        output_tokens_per_s=output_tokens_per_s,
    )


__all__ = ["LoadTestConfig", "LoadTestResult", "run_load_test"]
