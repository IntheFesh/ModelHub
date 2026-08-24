"""Unit tests for train/grpo/reward.py."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.eval.records import PredictionRecord
from modelhub.train.grpo.reward import MaskReason, RewardOutcome, RolloutReward, compute_reward


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


class TestRolloutRewardInvariant:
    def test_masked_without_mask_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="MASKED outcome must carry"):
            RolloutReward(
                sample_id="q0", reward=None, outcome=RewardOutcome.MASKED, mask_reason=None
            )

    def test_masked_with_a_real_reward_rejected(self) -> None:
        with pytest.raises(ValueError, match="MASKED outcome must carry"):
            RolloutReward(
                sample_id="q0",
                reward=0.0,
                outcome=RewardOutcome.MASKED,
                mask_reason=MaskReason.HARNESS_ERROR,
            )

    def test_correct_without_a_reward_rejected(self) -> None:
        with pytest.raises(ValueError, match="CORRECT outcome must carry"):
            RolloutReward(
                sample_id="q0", reward=None, outcome=RewardOutcome.CORRECT, mask_reason=None
            )

    def test_correct_with_a_mask_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="CORRECT outcome must carry"):
            RolloutReward(
                sample_id="q0",
                reward=1.0,
                outcome=RewardOutcome.CORRECT,
                mask_reason=MaskReason.HARNESS_ERROR,
            )


class TestComputeReward:
    def test_exec_ok_and_equal_is_correct(self) -> None:
        r = compute_reward(_prediction())
        assert r.reward == 1.0
        assert r.outcome is RewardOutcome.CORRECT
        assert r.mask_reason is None

    def test_exec_ok_and_not_equal_is_incorrect_zero(self) -> None:
        r = compute_reward(_prediction(comparison_result=ComparisonResult.NOT_EQUAL))
        assert r.reward == 0.0
        assert r.outcome is RewardOutcome.INCORRECT

    @pytest.mark.parametrize("exec_code", [ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT])
    def test_syntax_semantic_timeout_are_incorrect_zero(self, exec_code: ErrorCode) -> None:
        r = compute_reward(_prediction(exec_code=exec_code, comparison_result=None))
        assert r.reward == 0.0
        assert r.outcome is RewardOutcome.INCORRECT

    def test_unsafe_statement_is_incorrect_zero(self) -> None:
        r = compute_reward(
            _prediction(exec_code=ErrorCode.UNSAFE_STATEMENT, comparison_result=None)
        )
        assert r.reward == 0.0
        assert r.outcome is RewardOutcome.INCORRECT

    def test_harness_db_unavailable_is_masked_not_zero(self) -> None:
        r = compute_reward(
            _prediction(exec_code=ErrorCode.HARNESS_DB_UNAVAILABLE, comparison_result=None)
        )
        assert r.reward is None
        assert r.outcome is RewardOutcome.MASKED
        assert r.mask_reason is MaskReason.HARNESS_ERROR

    def test_harness_internal_is_masked_not_zero(self) -> None:
        r = compute_reward(
            _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
        )
        assert r.outcome is RewardOutcome.MASKED
        assert r.mask_reason is MaskReason.HARNESS_ERROR

    def test_output_truncated_is_masked_not_zero(self) -> None:
        r = compute_reward(
            _prediction(
                finish_reason="length", exec_code=ErrorCode.OUTPUT_TRUNCATED, comparison_result=None
            )
        )
        assert r.reward is None
        assert r.mask_reason is MaskReason.OUTPUT_TRUNCATED

    def test_undecidable_is_masked_not_zero(self) -> None:
        r = compute_reward(
            _prediction(
                comparison_result=ComparisonResult.UNDECIDABLE,
                undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
            )
        )
        assert r.reward is None
        assert r.mask_reason is MaskReason.UNDECIDABLE

    def test_unhandled_exec_code_raises(self) -> None:
        with pytest.raises(ValueError, match="unhandled exec_code"):
            compute_reward(_prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None))
