"""Scrape a live vLLM instance's own Prometheus `/metrics` endpoint.

Prefix-cache hit rate (group 2) and speculative-decode acceptance rate
(group 3) are not numbers this project can compute from the client side —
they live inside vLLM's own counters. This module fetches the raw
Prometheus text-exposition output over HTTP and pulls out named counters;
it does not try to be a general Prometheus client.

★ vLLM's own metric names have moved across versions (the
`vllm:gpu_prefix_cache_*` / `vllm:prefix_cache_*` rename being the
concrete example this project has actually hit). `fetch_counter` takes
the metric name as a caller-supplied argument rather than hardcoding one
inside this module, and `configs/bench/experiments/*.yaml` records
exactly which name string each experiment was run against — so a
version bump shows up as a config diff to review, not a silent
zero-vs-missing ambiguity (CLAUDE.md §1.1: missing is `None`, not 0).
"""

from __future__ import annotations

import re

_METRICS_TIMEOUT_S = 10.0
# `name{labels} value` or `name value` — the Prometheus text-exposition
# format (https://prometheus.io/docs/instrumenting/exposition_formats/).
_METRIC_LINE_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(?P<value>\S+)$"
)


def parse_prometheus_counter(metrics_text: str, metric_name: str) -> float | None:
    """Sum every sample line whose metric name matches `metric_name`
    (a metric can appear once per label combination — vLLM's counters
    are usually unlabeled, but this stays correct if that ever changes).
    Returns `None` if the metric name never appears — a genuinely
    missing counter, not a fabricated 0.0 (CLAUDE.md §1.1)."""
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_PATTERN.match(line)
        if match is None or match.group("name") != metric_name:
            continue
        try:
            total += float(match.group("value"))
        except ValueError:
            continue
        found = True
    return total if found else None


def fetch_metrics_text(metrics_url: str) -> str:
    """Real HTTP GET against a live vLLM instance's `/metrics` endpoint.
    Requires `httpx` (the `serve` extra) and a reachable server — there is
    no fallback path; a caller with no live server to hit should not call
    this function at all (see each experiment group's own
    `requires_gpu`-marked tests, not a mocked response here)."""
    import httpx

    response = httpx.get(metrics_url, timeout=_METRICS_TIMEOUT_S)
    response.raise_for_status()
    return response.text


def fetch_counter(metrics_url: str, metric_name: str) -> float | None:
    return parse_prometheus_counter(fetch_metrics_text(metrics_url), metric_name)


__all__ = ["fetch_counter", "fetch_metrics_text", "parse_prometheus_counter"]
