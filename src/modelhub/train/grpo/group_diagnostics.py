"""PLAN.md ★ item 5: "reward 全 0 或组内全相同 → 告警。GRPO 的 advantage
是组内归一化的，组内 reward 全相同时 advantage 恒为 0，这一组白跑了".

A GRPO "group" is the k rollouts sampled for one prompt — advantage is
computed by normalizing reward within the group, so a group whose scored
rewards are all identical (including the all-0.0 case PLAN.md names
specifically, but not only that case — all-1.0 is exactly as degenerate)
produces a zero-variance advantage signal for every sample in it. A group
where every rollout got masked is the same problem from a different
cause: there is nothing left to normalize.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.train.grpo.reward import RolloutReward


@dataclass(frozen=True)
class RolloutGroup:
    question_id: str
    rewards: tuple[RolloutReward, ...]

    def __post_init__(self) -> None:
        if not self.rewards:
            raise ValueError(f"{self.question_id}: a rollout group must have at least one rollout")
        mismatched = [r for r in self.rewards if r.sample_id != self.question_id]
        if mismatched:
            raise ValueError(
                f"{self.question_id}: every RolloutReward in a group must share the "
                f"group's question_id as its sample_id — got {[r.sample_id for r in mismatched]}"
            )


@dataclass(frozen=True)
class GroupDiagnostics:
    question_id: str
    total_rollouts: int
    masked_count: int
    scored_rewards: tuple[float, ...]
    is_degenerate: bool
    degenerate_reason: str | None


def diagnose_group(group: RolloutGroup) -> GroupDiagnostics:
    scored = tuple(r.reward for r in group.rewards if r.reward is not None)
    masked_count = len(group.rewards) - len(scored)

    if not scored:
        return GroupDiagnostics(
            question_id=group.question_id,
            total_rollouts=len(group.rewards),
            masked_count=masked_count,
            scored_rewards=scored,
            is_degenerate=True,
            degenerate_reason="every rollout in this group was masked — nothing to normalize",
        )
    if len(set(scored)) <= 1:
        return GroupDiagnostics(
            question_id=group.question_id,
            total_rollouts=len(group.rewards),
            masked_count=masked_count,
            scored_rewards=scored,
            is_degenerate=True,
            degenerate_reason=(
                f"all {len(scored)} scored reward(s) are identical ({scored[0]}) — "
                f"advantage would be zero for every rollout in this group"
            ),
        )
    return GroupDiagnostics(
        question_id=group.question_id,
        total_rollouts=len(group.rewards),
        masked_count=masked_count,
        scored_rewards=scored,
        is_degenerate=False,
        degenerate_reason=None,
    )


__all__ = ["GroupDiagnostics", "RolloutGroup", "diagnose_group"]
