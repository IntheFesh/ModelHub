"""Gate 2/5: regression detection — CLAUDE.md §1.5's required meta-test
`test_regression_detector_fires` covers exactly this gate ("新模型在老
模型答对的 case 上答错 → 回归检测触发").

A candidate can pass the absolute accuracy floor while still being worse
than the currently-deployed model on specific cases the old model
handled correctly — silently shipping that regression is exactly the
"looks fine in aggregate" failure mode CLAUDE.md §1 exists to prevent.
This gate compares sample-by-sample, not just headline accuracy deltas.
"""

from __future__ import annotations

from modelhub.common.config import ModelHubBaseConfig
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.records import PredictionRecord
from modelhub.gate.types import GateDecision, GateResult

_GATE_NAME = "regression"


class RegressionGateConfig(ModelHubBaseConfig):
    max_regressions: int


def find_regressions(
    baseline_predictions: list[PredictionRecord], candidate_predictions: list[PredictionRecord]
) -> list[str]:
    """`sample_id`s the baseline answered correctly (`EQUAL`) that the
    candidate does not — regardless of *why* the candidate got it wrong
    (`NOT_EQUAL`, a fresh `SYNTAX`/`SEMANTIC` error, `OUTPUT_TRUNCATED`,
    ...). A sample only present in one of the two runs is not comparable
    and is skipped, not treated as a regression."""
    baseline_by_id = {r.sample_id: r for r in baseline_predictions}
    regressions = []
    for candidate in candidate_predictions:
        baseline = baseline_by_id.get(candidate.sample_id)
        if baseline is None:
            continue
        if (
            baseline.comparison_result is ComparisonResult.EQUAL
            and candidate.comparison_result is not ComparisonResult.EQUAL
        ):
            regressions.append(candidate.sample_id)
    return regressions


def check_regression_gate(
    baseline_predictions: list[PredictionRecord] | None,
    candidate_predictions: list[PredictionRecord],
    config: RegressionGateConfig,
) -> GateResult:
    if baseline_predictions is None:
        return GateResult(
            _GATE_NAME,
            GateDecision.NOT_APPLICABLE,
            "no baseline predictions supplied — nothing to regress against "
            "(expected only for the first model ever admitted)",
            {},
        )

    regressions = find_regressions(baseline_predictions, candidate_predictions)
    if len(regressions) > config.max_regressions:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            f"{len(regressions)} regression(s) exceed the allowed "
            f"{config.max_regressions}: {regressions[:10]}",
            {"regression_count": len(regressions), "regressed_sample_ids": regressions},
        )
    return GateResult(
        _GATE_NAME,
        GateDecision.PASS,
        f"{len(regressions)} regression(s), within the allowed {config.max_regressions}",
        {"regression_count": len(regressions), "regressed_sample_ids": regressions},
    )


__all__ = ["RegressionGateConfig", "check_regression_gate", "find_regressions"]
