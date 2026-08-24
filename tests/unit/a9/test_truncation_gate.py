"""Unit tests for gate/truncation_gate.py."""

from __future__ import annotations

from tests.unit.a9.fakes import prediction

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import compute_metrics
from modelhub.gate.truncation_gate import TruncationGateConfig, check_truncation_gate
from modelhub.gate.types import GateDecision


def _metrics(*, truncated: int, total: int):
    records = [
        prediction(
            f"t{i}",
            exec_code=ErrorCode.OUTPUT_TRUNCATED,
            comparison_result=None,
            finish_reason="length",
        )
        for i in range(truncated)
    ]
    records += [
        prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL)
        for i in range(total - truncated)
    ]
    return compute_metrics(records)


def test_low_truncation_rate_passes() -> None:
    metrics = _metrics(truncated=1, total=100)
    result = check_truncation_gate(metrics, TruncationGateConfig(max_output_truncated_rate=0.05))
    assert result.decision is GateDecision.PASS


def test_high_truncation_rate_rejects() -> None:
    metrics = _metrics(truncated=20, total=100)
    result = check_truncation_gate(metrics, TruncationGateConfig(max_output_truncated_rate=0.05))
    assert result.decision is GateDecision.REJECT


def test_exactly_at_threshold_passes() -> None:
    metrics = _metrics(truncated=5, total=100)
    result = check_truncation_gate(metrics, TruncationGateConfig(max_output_truncated_rate=0.05))
    assert result.decision is GateDecision.PASS
