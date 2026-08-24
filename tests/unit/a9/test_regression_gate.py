"""Unit tests for gate/regression_gate.py — this is the gate CLAUDE.md
§1.5's required meta-test `test_regression_detector_fires` covers; see
tests/meta/a9/ for that dedicated meta-test."""

from __future__ import annotations

from tests.unit.a9.fakes import prediction

from modelhub.compare.result_types import ComparisonResult
from modelhub.gate.regression_gate import (
    RegressionGateConfig,
    check_regression_gate,
    find_regressions,
)
from modelhub.gate.types import GateDecision


def test_find_regressions_detects_correct_to_incorrect() -> None:
    baseline = [prediction("s1", comparison_result=ComparisonResult.EQUAL)]
    candidate = [prediction("s1", comparison_result=ComparisonResult.NOT_EQUAL)]
    assert find_regressions(baseline, candidate) == ["s1"]


def test_find_regressions_ignores_improvements() -> None:
    baseline = [prediction("s1", comparison_result=ComparisonResult.NOT_EQUAL)]
    candidate = [prediction("s1", comparison_result=ComparisonResult.EQUAL)]
    assert find_regressions(baseline, candidate) == []


def test_find_regressions_ignores_samples_missing_from_baseline() -> None:
    baseline: list = []
    candidate = [prediction("s1", comparison_result=ComparisonResult.NOT_EQUAL)]
    assert find_regressions(baseline, candidate) == []


def test_find_regressions_ignores_samples_still_correct() -> None:
    baseline = [prediction("s1", comparison_result=ComparisonResult.EQUAL)]
    candidate = [prediction("s1", comparison_result=ComparisonResult.EQUAL)]
    assert find_regressions(baseline, candidate) == []


def test_check_regression_gate_no_baseline_is_not_applicable() -> None:
    candidate = [prediction("s1", comparison_result=ComparisonResult.EQUAL)]
    result = check_regression_gate(None, candidate, RegressionGateConfig(max_regressions=0))
    assert result.decision is GateDecision.NOT_APPLICABLE


def test_check_regression_gate_within_budget_passes() -> None:
    baseline = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(5)]
    candidate = [
        prediction("s0", comparison_result=ComparisonResult.NOT_EQUAL),
        *[prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(1, 5)],
    ]
    result = check_regression_gate(baseline, candidate, RegressionGateConfig(max_regressions=1))
    assert result.decision is GateDecision.PASS
    assert result.metrics["regression_count"] == 1


def test_check_regression_gate_over_budget_rejects() -> None:
    baseline = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(5)]
    candidate = [
        prediction(f"s{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(5)
    ]
    result = check_regression_gate(baseline, candidate, RegressionGateConfig(max_regressions=1))
    assert result.decision is GateDecision.REJECT
    assert result.metrics["regression_count"] == 5
