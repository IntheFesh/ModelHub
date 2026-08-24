"""Unit tests for train/grpo/group_diagnostics.py."""

from __future__ import annotations

import pytest

from modelhub.train.grpo.group_diagnostics import RolloutGroup, diagnose_group
from modelhub.train.grpo.reward import MaskReason, RewardOutcome, RolloutReward


def _reward(**overrides: object) -> RolloutReward:
    defaults: dict[str, object] = {
        "sample_id": "q0",
        "reward": 1.0,
        "outcome": RewardOutcome.CORRECT,
        "mask_reason": None,
    }
    defaults.update(overrides)
    return RolloutReward(**defaults)  # type: ignore[arg-type]


class TestRolloutGroupInvariant:
    def test_empty_rewards_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one rollout"):
            RolloutGroup(question_id="q0", rewards=())

    def test_mismatched_sample_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="share the group's question_id"):
            RolloutGroup(question_id="q0", rewards=(_reward(sample_id="q1"),))


class TestDiagnoseGroup:
    def test_mixed_rewards_not_degenerate(self) -> None:
        group = RolloutGroup(
            question_id="q0",
            rewards=(
                _reward(reward=1.0, outcome=RewardOutcome.CORRECT),
                _reward(reward=0.0, outcome=RewardOutcome.INCORRECT),
            ),
        )
        diag = diagnose_group(group)
        assert diag.is_degenerate is False
        assert diag.scored_rewards == (1.0, 0.0)
        assert diag.masked_count == 0

    def test_all_zero_is_degenerate(self) -> None:
        group = RolloutGroup(
            question_id="q0",
            rewards=tuple(_reward(reward=0.0, outcome=RewardOutcome.INCORRECT) for _ in range(4)),
        )
        diag = diagnose_group(group)
        assert diag.is_degenerate is True
        assert "identical" in (diag.degenerate_reason or "")

    def test_all_one_is_also_degenerate(self) -> None:
        """PLAN.md names "全0" explicitly but all-1.0 is exactly as
        degenerate — the check must not be a literal-zero special case."""
        group = RolloutGroup(
            question_id="q0",
            rewards=tuple(_reward(reward=1.0, outcome=RewardOutcome.CORRECT) for _ in range(4)),
        )
        diag = diagnose_group(group)
        assert diag.is_degenerate is True

    def test_all_masked_is_degenerate(self) -> None:
        group = RolloutGroup(
            question_id="q0",
            rewards=tuple(
                _reward(
                    reward=None,
                    outcome=RewardOutcome.MASKED,
                    mask_reason=MaskReason.HARNESS_ERROR,
                )
                for _ in range(4)
            ),
        )
        diag = diagnose_group(group)
        assert diag.is_degenerate is True
        assert diag.masked_count == 4
        assert diag.scored_rewards == ()
        assert "nothing to normalize" in (diag.degenerate_reason or "")

    def test_masked_plus_mixed_scored_not_degenerate(self) -> None:
        group = RolloutGroup(
            question_id="q0",
            rewards=(
                _reward(
                    reward=None,
                    outcome=RewardOutcome.MASKED,
                    mask_reason=MaskReason.HARNESS_ERROR,
                ),
                _reward(reward=1.0, outcome=RewardOutcome.CORRECT),
                _reward(reward=0.0, outcome=RewardOutcome.INCORRECT),
            ),
        )
        diag = diagnose_group(group)
        assert diag.is_degenerate is False
        assert diag.masked_count == 1
