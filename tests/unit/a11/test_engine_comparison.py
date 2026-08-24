"""Unit tests for bench/experiments/engine_comparison.py."""

from __future__ import annotations

from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.experiments.engine_comparison import (
    EngineComparisonExperimentConfig,
    run_engine_comparison,
)


def _config(**overrides: object) -> EngineComparisonExperimentConfig:
    defaults: dict[str, object] = {
        "concurrency": 4,
        "num_requests": 10,
        "max_tokens": 64,
        "temperature": 0.0,
        "generate_timeout_s": 5.0,
        "prompt_length_label": "1k",
        "sla_latency_p99_s": 1.0,
    }
    defaults.update(overrides)
    return EngineComparisonExperimentConfig.model_validate(defaults)


def test_both_engines_measured_under_identical_config() -> None:
    result = run_engine_comparison(
        vllm_client=BenchFakeClient(),
        sglang_client=BenchFakeClient(),
        prompt="SELECT ...",
        config=_config(),
    )
    assert result.vllm_result.num_requests == 10
    assert result.sglang_result.num_requests == 10
    assert result.vllm_within_sla is True
    assert result.sglang_within_sla is True


def test_qps_ratio_reflects_relative_speed() -> None:
    result = run_engine_comparison(
        vllm_client=BenchFakeClient(latency_s=0.02),
        sglang_client=BenchFakeClient(latency_s=0.0),
        prompt="SELECT ...",
        config=_config(concurrency=1, num_requests=5),
    )
    assert result.qps_ratio_sglang_over_vllm is not None
    assert result.qps_ratio_sglang_over_vllm > 1.0


def test_sla_flag_none_when_no_successful_requests() -> None:
    result = run_engine_comparison(
        vllm_client=BenchFakeClient(fail_every_n=1),
        sglang_client=BenchFakeClient(),
        prompt="SELECT ...",
        config=_config(num_requests=3),
    )
    assert result.vllm_result.latency_p99_s is None
    assert result.vllm_within_sla is None
