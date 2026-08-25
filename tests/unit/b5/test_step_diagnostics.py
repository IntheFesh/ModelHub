"""Unit tests for train/grpo/step_diagnostics.py."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.records import PredictionRecord
from modelhub.train.grpo.group_diagnostics import GroupDiagnostics
from modelhub.train.grpo.step_diagnostics import (
    assert_harness_error_rate_ok,
    assert_unclassified_rate_ok,
    compute_step_diagnostics,
)


def _prediction(**overrides: object) -> PredictionRecord:
    defaults: dict[str, object] = {
        "sample_id": "q0",
        "db_id": "school",
        "difficulty": "simple",
        "predicted_sql": "SELECT 1",
        "finish_reason": "stop",
        "exec_code": ErrorCode.EXEC_OK,
        "comparison_result": ComparisonResult.EQUAL,
        "elapsed_s": 0.01,
    }
    defaults.update(overrides)
    return PredictionRecord.model_validate(defaults)


def _group_diag(*, is_degenerate: bool) -> GroupDiagnostics:
    return GroupDiagnostics(
        question_id="q0",
        total_rollouts=4,
        masked_count=0,
        scored_rewards=(1.0, 0.0),
        is_degenerate=is_degenerate,
        degenerate_reason="x" if is_degenerate else None,
    )


class TestComputeStepDiagnostics:
    def test_rates_computed_correctly(self) -> None:
        predictions = (
            [_prediction()] * 10
            + [_prediction(exec_code=ErrorCode.TIMEOUT, comparison_result=None)] * 2
            + [
                _prediction(
                    finish_reason="length",
                    exec_code=ErrorCode.OUTPUT_TRUNCATED,
                    comparison_result=None,
                )
            ]
            * 3
            + [_prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)] * 1
        )
        diag = compute_step_diagnostics(5, predictions, [_group_diag(is_degenerate=False)])
        assert diag.total_rollouts == 16
        assert diag.timeout_count == 2
        assert diag.truncated_count == 3
        assert diag.harness_error_count == 1
        assert diag.timeout_rate == pytest.approx(2 / 16)
        assert diag.harness_error_rate == pytest.approx(1 / 16)

    def test_unclassified_count_and_rate_tracked_separately_from_harness(self) -> None:
        # Regression test: UNCLASSIFIED used to be invisible everywhere in
        # this module (PredictionRecord.is_harness_error doesn't cover it).
        predictions = [_prediction()] * 8 + [
            _prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None)
        ] * 2
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        assert diag.unclassified_count == 2
        assert diag.unclassified_rate == pytest.approx(0.2)
        assert diag.harness_error_count == 0  # not double-counted as a harness error

    def test_degenerate_group_rate(self) -> None:
        diag = compute_step_diagnostics(
            1, [_prediction()], [_group_diag(is_degenerate=True), _group_diag(is_degenerate=False)]
        )
        assert diag.degenerate_group_count == 1
        assert diag.total_group_count == 2
        assert diag.degenerate_group_rate == pytest.approx(0.5)

    def test_empty_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="zero predictions"):
            compute_step_diagnostics(1, [], [_group_diag(is_degenerate=False)])

    def test_empty_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="zero groups"):
            compute_step_diagnostics(1, [_prediction()], [])


class TestAssertHarnessErrorRateOk:
    def test_below_threshold_does_not_raise(self) -> None:
        predictions = [_prediction()] * 99 + [
            _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        ]
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        assert_harness_error_rate_ok(diag)  # 1/100 == threshold, must not raise

    def test_above_threshold_raises_and_aborts(self) -> None:
        predictions = [_prediction()] * 90 + [
            _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        ] * 10
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        with pytest.raises(ValueError, match="aborting GRPO training"):
            assert_harness_error_rate_ok(diag)

    def test_custom_threshold_respected(self) -> None:
        predictions = [_prediction()] * 8 + [
            _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        ] * 2
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        assert_harness_error_rate_ok(diag, threshold=0.5)  # 20% <= 50%, must not raise
        with pytest.raises(ValueError):
            assert_harness_error_rate_ok(diag, threshold=0.1)  # 20% > 10%


class TestAssertUnclassifiedRateOk:
    def test_below_threshold_does_not_raise(self) -> None:
        predictions = [_prediction()] * 99 + [
            _prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None)
        ]
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        assert_unclassified_rate_ok(diag)  # 1/100 == threshold, must not raise

    def test_above_threshold_raises_and_aborts(self) -> None:
        predictions = [_prediction()] * 90 + [
            _prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None)
        ] * 10
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        with pytest.raises(ValueError, match="aborting GRPO training"):
            assert_unclassified_rate_ok(diag)

    def test_a_pure_harness_flood_does_not_trip_the_unclassified_guard(self) -> None:
        # The two guards are independent: a harness-only flood must not be
        # silently absorbed into (or hidden by) the unclassified checker.
        predictions = [_prediction()] * 90 + [
            _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        ] * 10
        diag = compute_step_diagnostics(1, predictions, [_group_diag(is_degenerate=False)])
        assert_unclassified_rate_ok(diag)  # must not raise: 0% unclassified
