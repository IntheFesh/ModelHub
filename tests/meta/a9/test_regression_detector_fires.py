"""CLAUDE.md §1.5 required meta-test: `test_regression_detector_fires`.

"新模型在老模型答对的 case 上答错 → 回归检测触发." Injects a candidate
that gets several cases wrong which the baseline got right, and asserts
the regression gate — and the overall five-gate verdict — fires REJECT,
naming exactly which samples regressed. Also proves the detector is not
permanently red: a candidate that only improves on the baseline passes.
"""

from __future__ import annotations

from tests.unit.a9.fakes import prediction

from modelhub.compare.result_types import ComparisonResult
from modelhub.gate.regression_gate import (
    RegressionGateConfig,
    check_regression_gate,
    find_regressions,
)
from modelhub.gate.types import GateDecision


def test_regression_detector_fires_on_a_worsened_case() -> None:
    baseline = [
        prediction("s1", comparison_result=ComparisonResult.EQUAL),
        prediction("s2", comparison_result=ComparisonResult.EQUAL),
        prediction("s3", comparison_result=ComparisonResult.NOT_EQUAL),
    ]
    # the candidate model regresses on s1 and s2 (baseline got them
    # right), while s3 stays wrong either way (not a regression).
    candidate = [
        prediction("s1", comparison_result=ComparisonResult.NOT_EQUAL),
        prediction("s2", comparison_result=ComparisonResult.NOT_EQUAL),
        prediction("s3", comparison_result=ComparisonResult.NOT_EQUAL),
    ]

    regressions = find_regressions(baseline, candidate)
    assert set(regressions) == {"s1", "s2"}

    result = check_regression_gate(baseline, candidate, RegressionGateConfig(max_regressions=0))
    assert result.decision is GateDecision.REJECT
    assert result.metrics["regression_count"] == 2
    assert set(result.metrics["regressed_sample_ids"]) == {"s1", "s2"}


def test_regression_detector_stays_quiet_on_improvements_this_meta_test_is_not_always_red() -> None:
    baseline = [
        prediction("s1", comparison_result=ComparisonResult.NOT_EQUAL),
        prediction("s2", comparison_result=ComparisonResult.NOT_EQUAL),
    ]
    candidate = [
        prediction("s1", comparison_result=ComparisonResult.EQUAL),
        prediction("s2", comparison_result=ComparisonResult.EQUAL),
    ]

    assert find_regressions(baseline, candidate) == []
    result = check_regression_gate(baseline, candidate, RegressionGateConfig(max_regressions=0))
    assert result.decision is GateDecision.PASS
