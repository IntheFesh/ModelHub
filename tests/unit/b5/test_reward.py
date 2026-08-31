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

    def test_unclassified_is_masked_not_zero(self) -> None:
        # Regression test: ErrorCode.UNCLASSIFIED is a real value every
        # sqlexec backend's exception classifier can emit as its fallback
        # (backends.py's sqlite/duckdb/postgres classifiers) — compute_
        # reward must not crash on it, and must not silently score it 0.0
        # either (fault attribution is unknown, so scoring it INCORRECT
        # would risk exactly the "system error taught as model error"
        # hazard this module's own docstring calls out as CLAUDE.md §2.3's
        # single biggest warning).
        r = compute_reward(_prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None))
        assert r.reward is None
        assert r.outcome is RewardOutcome.MASKED
        assert r.mask_reason is MaskReason.UNCLASSIFIED

    def test_genuinely_unhandled_exec_code_still_raises(self) -> None:
        # Proves the exhaustiveness guard itself still works now that
        # UNCLASSIFIED has a real branch — RUN_POLLUTED is a manifest-
        # pollution code that can never legitimately land in
        # PredictionRecord.exec_code, a safe "truly unhandled" sentinel.
        with pytest.raises(ValueError, match="unhandled exec_code"):
            compute_reward(_prediction(exec_code=ErrorCode.RUN_POLLUTED, comparison_result=None))
