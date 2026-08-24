"""Group 3: speculative decoding, dual-path (PLAN.md v2's major rewrite).

Qwen3.5-9B has a native MTP head (trained with multi-step prediction) —
the draft path matched to the model, used directly. Arctic-7B has no
matching EAGLE head (training one is a separate project's scope), so its
only viable path is n-gram/suffix matching — which this project's SQL
workload happens to favor, since generated SQL heavily copies schema
identifiers verbatim out of the prompt.

Both paths are measured the same way: speedup ratio and acceptance rate
at each concurrency level, and the resulting decay curve — PLAN.md's
expected shape is that the benefit shrinks and can cross below 1.0 as
concurrency rises (the batch is already compute-saturated, so draft-token
computation competes with real work instead of hiding behind idle
capacity).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelhub.bench.experiments.vllm_metrics import fetch_counter
from modelhub.bench.load_test import LoadTestConfig, run_load_test
from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.model_client import ModelClient


class SpeculativeDecodingPath(StrEnum):
    MTP = "MTP"  # Qwen3.5-9B's native multi-token-prediction head
    NGRAM = "NGRAM"  # Arctic-7B: no matching EAGLE head, n-gram/suffix instead


class SpeculativeDecodingExperimentConfig(ModelHubBaseConfig):
    metrics_url: str
    accepted_tokens_metric_name: str
    draft_tokens_metric_name: str
    concurrency_levels: list[int]
    num_requests_per_level: int
    max_tokens: int
    temperature: float
    generate_timeout_s: float
    prompt_length_label: str


@dataclass(frozen=True)
class SpeculativeAcceptanceStats:
    accepted_tokens: int
    draft_tokens: int

    @property
    def acceptance_rate(self) -> float | None:
        if self.draft_tokens == 0:
            return None
        return self.accepted_tokens / self.draft_tokens


def measure_acceptance_rate(
    *, metrics_url: str, accepted_tokens_metric_name: str, draft_tokens_metric_name: str
) -> SpeculativeAcceptanceStats:
    """Real vLLM speculative-decode counters only — an acceptance rate is
    not something the client side can estimate from wall-clock timing
    alone (CLAUDE.md §1.1: missing is not 0)."""
    accepted = fetch_counter(metrics_url, accepted_tokens_metric_name)
    draft = fetch_counter(metrics_url, draft_tokens_metric_name)
    if accepted is None or draft is None:
        raise ValueError(
            f"vLLM did not report speculative-decode counters "
            f"{accepted_tokens_metric_name!r}/{draft_tokens_metric_name!r} at "
            f"{metrics_url} — refusing to compute an acceptance rate from a "
            f"missing metric"
        )
    return SpeculativeAcceptanceStats(int(accepted), int(draft))


@dataclass(frozen=True)
class ConcurrencySpeedupPoint:
    concurrency: int
    baseline_qps: float
    speculative_qps: float

    @property
    def speedup_ratio(self) -> float | None:
        if self.baseline_qps == 0:
            return None
        return self.speculative_qps / self.baseline_qps


@dataclass(frozen=True)
class SpeculativeDecayCurve:
    path: SpeculativeDecodingPath
    points: tuple[ConcurrencySpeedupPoint, ...]

    @property
    def collapses_under_concurrency(self) -> bool:
        """True iff `speedup_ratio` drops to `<= 1.0` (no benefit, or a
        net loss) at any measured concurrency level — PLAN.md's expected
        shape: "高并发下 batch 已填满，draft 计算反挤占算力，收益归零甚至
        为负". States the comparison; a real curve that never collapses
        within the configured concurrency range legitimately returns
        False, that is not this property being wrong."""
        ratios = [p.speedup_ratio for p in self.points if p.speedup_ratio is not None]
        if not ratios:
            raise ValueError(
                "cannot judge decay with zero valid speedup points (all baseline_qps=0)"
            )
        return any(ratio <= 1.0 for ratio in ratios)


def measure_concurrency_decay_curve(
    path: SpeculativeDecodingPath,
    *,
    baseline_client: ModelClient,
    speculative_client: ModelClient,
    prompt: str,
    config: SpeculativeDecodingExperimentConfig,
) -> SpeculativeDecayCurve:
    """`baseline_client`/`speculative_client` are the same served model
    with speculative decoding off/on respectively (a serving-side flag
    this function does not set) — at each configured concurrency level,
    a real `run_load_test` (A8) pass is run against each, and the QPS
    ratio at that level becomes one point on the curve."""
    points = []
    for concurrency in config.concurrency_levels:
        load_config = LoadTestConfig(
            concurrency=concurrency,
            num_requests=config.num_requests_per_level,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            generate_timeout_s=config.generate_timeout_s,
            prompt_length_label=config.prompt_length_label,
        )
        baseline_result = run_load_test(baseline_client, prompt, load_config)
        speculative_result = run_load_test(speculative_client, prompt, load_config)
        points.append(
            ConcurrencySpeedupPoint(
                concurrency=concurrency,
                baseline_qps=baseline_result.qps,
                speculative_qps=speculative_result.qps,
            )
        )
    return SpeculativeDecayCurve(path=path, points=tuple(points))


__all__ = [
    "ConcurrencySpeedupPoint",
    "SpeculativeAcceptanceStats",
    "SpeculativeDecayCurve",
    "SpeculativeDecodingExperimentConfig",
    "SpeculativeDecodingPath",
    "measure_acceptance_rate",
    "measure_concurrency_decay_curve",
]
