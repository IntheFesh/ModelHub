"""Unit tests for bench/experiments/vllm_metrics.py: Prometheus
text-exposition parsing (real parsing logic — the HTTP fetch itself is
requires_gpu/requires_network territory, tested separately)."""

from __future__ import annotations

from modelhub.bench.experiments.vllm_metrics import parse_prometheus_counter

_SAMPLE_METRICS = """\
# HELP vllm:gpu_prefix_cache_hits_total Prefix cache hits.
# TYPE vllm:gpu_prefix_cache_hits_total counter
vllm:gpu_prefix_cache_hits_total 123.0
# HELP vllm:gpu_prefix_cache_queries_total Prefix cache queries.
# TYPE vllm:gpu_prefix_cache_queries_total counter
vllm:gpu_prefix_cache_queries_total 456.0
vllm:num_requests_running{model_name="arctic-7b"} 3.0
"""


def test_parses_a_simple_unlabeled_counter() -> None:
    assert parse_prometheus_counter(_SAMPLE_METRICS, "vllm:gpu_prefix_cache_hits_total") == 123.0


def test_parses_a_labeled_metric() -> None:
    assert parse_prometheus_counter(_SAMPLE_METRICS, "vllm:num_requests_running") == 3.0


def test_missing_metric_returns_none_not_zero() -> None:
    assert parse_prometheus_counter(_SAMPLE_METRICS, "vllm:does_not_exist") is None


def test_ignores_comment_and_blank_lines() -> None:
    text = "# a comment\n\nvllm:x 5.0\n"
    assert parse_prometheus_counter(text, "vllm:x") == 5.0


def test_sums_multiple_label_combinations_for_the_same_metric_name() -> None:
    text = 'vllm:hits{model="a"} 10.0\nvllm:hits{model="b"} 5.0\n'
    assert parse_prometheus_counter(text, "vllm:hits") == 15.0


def test_empty_text_returns_none() -> None:
    assert parse_prometheus_counter("", "vllm:anything") is None
