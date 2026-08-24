"""A8 smoke test: load test -> capacity planning -> cost model, wired
end to end at a tiny scale against a fake backend (no GPU in this
sandbox to bench for real)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.capacity_planning import (
    CapacityPlanningConfig,
    estimate_max_concurrent_requests,
)
from modelhub.bench.cost_model import estimate_request_cost_usd
from modelhub.bench.load_test import LoadTestConfig, run_load_test
from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.serve.model_profile import load_model_profile

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_bench_pipeline_smoke() -> None:
    # 1. load test against a fake backend
    client = BenchFakeClient(prompt_tokens=150, completion_tokens=60)
    config = LoadTestConfig(
        concurrency=4,
        num_requests=8,
        max_tokens=64,
        temperature=0.0,
        generate_timeout_s=5.0,
        prompt_length_label="1k",
    )
    load_result = run_load_test(client, "x" * 400, config)
    assert load_result.succeeded == 8
    assert load_result.qps > 0
    assert load_result.output_tokens_per_s is not None

    # 2. capacity planning from the real committed model profile + config
    profile, _ = load_model_profile(
        _REPO_ROOT / "configs" / "serve" / "model_profiles" / "arctic_7b.yaml"
    )
    capacity_config, _ = load_yaml_config(
        CapacityPlanningConfig,
        _REPO_ROOT / "configs" / "bench" / "capacity_planning.yaml",
        stage=Stage.SERVE,
    )
    max_concurrency = estimate_max_concurrent_requests(
        profile,
        gpu_memory_bytes=capacity_config.gpu_memory_bytes,
        gpu_memory_utilization=capacity_config.gpu_memory_utilization,
        runtime_overhead_bytes=capacity_config.runtime_overhead_bytes,
        prompt_tokens=1024,
        max_new_tokens=capacity_config.max_new_tokens,
    )
    assert max_concurrency > 0

    # 3. cost model, fed by the load test's own measured throughput
    assert load_result.output_tokens_per_s is not None
    cost = estimate_request_cost_usd(
        prompt_tokens=load_result.total_prompt_tokens // load_result.succeeded,
        completion_tokens=load_result.total_completion_tokens // load_result.succeeded,
        gpu_hourly_cost_usd=3.6,  # illustrative — no confirmed 5090 rental rate exists yet
        prompt_tokens_per_s=load_result.output_tokens_per_s * 10,  # prefill is typically faster
        completion_tokens_per_s=load_result.output_tokens_per_s,
    )
    assert cost.cost_usd > 0
