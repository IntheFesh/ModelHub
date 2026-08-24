"""Unit tests for gate/accuracy_gate.py."""

from __future__ import annotations

from tests.unit.a9.fakes import prediction

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import compute_metrics
from modelhub.gate.accuracy_gate import AccuracyGateConfig, check_accuracy_gate
from modelhub.gate.types import GateDecision


def _config(min_accuracy: float = 0.5) -> AccuracyGateConfig:
    return AccuracyGateConfig(min_execution_accuracy=min_accuracy)


def test_accuracy_above_threshold_passes() -> None:
    records = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(6)]
    records += [prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(4)]
    metrics = compute_metrics(records)  # 60% accuracy
    result = check_accuracy_gate(metrics, _config(0.5))
    assert result.decision is GateDecision.PASS
    assert result.gate_name == "accuracy"


def test_accuracy_below_threshold_rejects() -> None:
    records = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(2)]
    records += [prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(8)]
    metrics = compute_metrics(records)  # 20% accuracy
    result = check_accuracy_gate(metrics, _config(0.5))
    assert result.decision is GateDecision.REJECT


def test_accuracy_exactly_at_threshold_passes() -> None:
    records = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(5)]
    records += [prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(5)]
    metrics = compute_metrics(records)  # exactly 50%
    result = check_accuracy_gate(metrics, _config(0.5))
    assert result.decision is GateDecision.PASS


def test_none_accuracy_rejects() -> None:
    # denominator == 0: every prediction excluded (all OUTPUT_TRUNCATED)
    records = [
        prediction(
            f"s{i}",
            exec_code=ErrorCode.OUTPUT_TRUNCATED,
            comparison_result=None,
            finish_reason="length",
        )
        for i in range(3)
    ]
    metrics = compute_metrics(records)
    assert metrics.execution_accuracy is None
    result = check_accuracy_gate(metrics, _config(0.0))
    assert result.decision is GateDecision.REJECT
