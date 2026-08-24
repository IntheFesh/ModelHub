"""PLAN.md's speculative-decoding expectation, in code: "高并发下 batch
已填满，draft 计算反挤占算力，收益归零甚至为负" — `SpeculativeDecayCurve.
collapses_under_concurrency` must be able to actually detect that
collapse, not just report False by construction.

Injects a speculative client that gets slower than the baseline at high
concurrency (simulating draft-token computation competing with real work
once the batch saturates) and asserts the curve reports a collapse.
Also proves the same check is not permanently red: a speculative client
that stays faster at every level reports no collapse.
"""

from __future__ import annotations

from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.experiments.speculative_decoding import (
    SpeculativeDecodingExperimentConfig,
    SpeculativeDecodingPath,
    measure_concurrency_decay_curve,
)


def _config(concurrency_levels: list[int]) -> SpeculativeDecodingExperimentConfig:
    return SpeculativeDecodingExperimentConfig.model_validate(
        {
            "metrics_url": "http://localhost:8000/metrics",
            "accepted_tokens_metric_name": "vllm:spec_decode_num_accepted_tokens_total",
            "draft_tokens_metric_name": "vllm:spec_decode_num_draft_tokens_total",
            "concurrency_levels": concurrency_levels,
            "num_requests_per_level": 8,
            "max_tokens": 32,
            "temperature": 0.0,
            "generate_timeout_s": 5.0,
            "prompt_length_label": "1k",
        }
    )


def test_a_saturated_high_concurrency_draft_path_collapses_the_curve() -> None:
    # baseline stays fast throughout; the "speculative" client is
    # deliberately made SLOWER than baseline — standing in for draft
    # computation stealing cycles from real work once the batch is full.
    curve = measure_concurrency_decay_curve(
        SpeculativeDecodingPath.MTP,
        baseline_client=BenchFakeClient(latency_s=0.0),
        speculative_client=BenchFakeClient(latency_s=0.03),
        prompt="SELECT * FROM students",
        config=_config([64]),
    )
    assert curve.collapses_under_concurrency is True


def test_this_check_is_not_always_red_a_genuinely_faster_path_does_not_collapse() -> None:
    curve = measure_concurrency_decay_curve(
        SpeculativeDecodingPath.NGRAM,
        baseline_client=BenchFakeClient(latency_s=0.02),
        speculative_client=BenchFakeClient(latency_s=0.0),
        prompt="SELECT * FROM students",
        config=_config([1, 4]),
    )
    assert curve.collapses_under_concurrency is False
