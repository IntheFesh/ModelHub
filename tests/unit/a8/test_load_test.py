"""Unit tests for bench/load_test.py."""

from __future__ import annotations

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.load_test import LoadTestConfig, _percentile, run_load_test


def _config(**overrides: object) -> LoadTestConfig:
    defaults: dict[str, object] = {
        "concurrency": 4,
        "num_requests": 8,
        "max_tokens": 64,
        "temperature": 0.0,
        "generate_timeout_s": 5.0,
        "prompt_length_label": "1k",
    }
    defaults.update(overrides)
    return LoadTestConfig.model_validate(defaults)


def test_percentile_basic_nearest_rank() -> None:
    sample = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(sample, 0) == 10.0
    assert _percentile(sample, 50) == 30.0
    assert _percentile(sample, 95) == 50.0
    assert _percentile(sample, 100) == 50.0


def test_percentile_single_element() -> None:
    assert _percentile([42.0], 50) == 42.0


def test_percentile_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        _percentile([], 50)


def test_percentile_rejects_out_of_range_pct() -> None:
    with pytest.raises(ValueError, match="pct"):
        _percentile([1.0], 101)


def test_all_requests_succeed() -> None:
    client = BenchFakeClient(prompt_tokens=100, completion_tokens=50)
    result = run_load_test(client, "prompt", _config(num_requests=6, concurrency=3))
    assert result.succeeded == 6
    assert result.failed == 0
    assert result.error_breakdown == {}
    assert result.latency_p50_s is not None
    assert result.latency_p95_s is not None
    assert result.latency_p99_s is not None
    assert result.total_prompt_tokens == 600
    assert result.total_completion_tokens == 300
    assert result.output_tokens_per_s is not None
    assert result.qps > 0


def test_failures_are_counted_and_classified() -> None:
    client = BenchFakeClient(fail_every_n=3)
    result = run_load_test(client, "prompt", _config(num_requests=9, concurrency=3))
    assert result.succeeded == 6
    assert result.failed == 3
    assert result.error_breakdown == {"RuntimeError": 3}


def test_prompt_length_label_is_carried_through() -> None:
    client = BenchFakeClient()
    result = run_load_test(client, "prompt", _config(prompt_length_label="8k", num_requests=1))
    assert result.prompt_length_label == "8k"


def test_higher_concurrency_is_meaningfully_faster_wall_clock() -> None:
    # A real (small) per-call latency proves concurrency actually
    # parallelizes requests instead of secretly running them serially —
    # same "prove it, don't assume it" spirit as A2's slow-task test.
    latency_s = 0.05
    num_requests = 10

    serial = run_load_test(
        BenchFakeClient(latency_s=latency_s),
        "prompt",
        _config(concurrency=1, num_requests=num_requests),
    )
    parallel = run_load_test(
        BenchFakeClient(latency_s=latency_s),
        "prompt",
        _config(concurrency=num_requests, num_requests=num_requests),
    )

    assert parallel.total_wall_time_s < serial.total_wall_time_s / 2


def test_output_tokens_per_s_is_none_when_all_requests_fail() -> None:
    client = BenchFakeClient(fail_every_n=1)  # every call fails
    result = run_load_test(client, "prompt", _config(num_requests=3, concurrency=3))
    assert result.succeeded == 0
    assert result.failed == 3
    assert result.output_tokens_per_s is None
    assert result.latency_p50_s is None
