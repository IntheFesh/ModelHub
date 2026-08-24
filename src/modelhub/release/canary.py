"""Canary traffic routing and staged rollout — PLAN.md's "灰度 5% →
注入劣化 → 自动回滚" drill: a new model version starts at a small
percentage of live traffic and only advances to the next stage once
`min_requests_per_stage` healthy requests have been observed at the
current one.

Routing is deterministic (hash of a request key modulo 100), not
`random.random()` — the same request key always resolves to the same
arm within a rollout. That is a real property canary routing needs (a
retried request, or a user's session, should not flip between stable and
canary from one call to the next), and it also makes routing decisions
reproducible in tests without seeding a global RNG.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from modelhub.common.config import ModelHubBaseConfig

Arm = Literal["stable", "canary"]


class CanaryConfig(ModelHubBaseConfig):
    # ascending percentages of traffic sent to the canary, ending at 100
    # (full rollout). CLAUDE.md doesn't mandate a specific ramp — PLAN.md
    # only fixes the *first* stage at 5%.
    stages: list[int]
    min_requests_per_stage: int

    def model_post_init(self, __context: object) -> None:
        if not self.stages:
            raise ValueError("stages must not be empty")
        if self.stages != sorted(self.stages):
            raise ValueError(f"stages must be ascending, got {self.stages}")
        if self.stages[-1] != 100:
            raise ValueError(f"the last stage must be 100 (full rollout), got {self.stages[-1]}")
        if any(not (0 < s <= 100) for s in self.stages):
            raise ValueError(f"every stage must be in (0, 100], got {self.stages}")


def route_canary_traffic(request_key: str, canary_percentage: int) -> Arm:
    if not 0 <= canary_percentage <= 100:
        raise ValueError(f"canary_percentage must be in [0, 100], got {canary_percentage}")
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100
    return "canary" if bucket < canary_percentage else "stable"


class RolloutStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class CanaryRolloutState:
    stage_index: int
    requests_at_current_stage: int
    status: RolloutStatus

    def current_percentage(self, config: CanaryConfig) -> int:
        return config.stages[self.stage_index]


def start_rollout() -> CanaryRolloutState:
    return CanaryRolloutState(
        stage_index=0, requests_at_current_stage=0, status=RolloutStatus.IN_PROGRESS
    )


def record_canary_request(state: CanaryRolloutState) -> CanaryRolloutState:
    if state.status is not RolloutStatus.IN_PROGRESS:
        raise ValueError(
            f"cannot record a request against a rollout with status {state.status.value}"
        )
    return CanaryRolloutState(
        stage_index=state.stage_index,
        requests_at_current_stage=state.requests_at_current_stage + 1,
        status=state.status,
    )


def try_advance_stage(state: CanaryRolloutState, config: CanaryConfig) -> CanaryRolloutState:
    """Advance to the next stage once `min_requests_per_stage` has been
    observed at the current one. No-op (returns `state` unchanged) if not
    enough requests have been observed yet, or the rollout isn't
    `IN_PROGRESS`."""
    if state.status is not RolloutStatus.IN_PROGRESS:
        return state
    if state.requests_at_current_stage < config.min_requests_per_stage:
        return state
    if state.stage_index == len(config.stages) - 1:
        return CanaryRolloutState(
            stage_index=state.stage_index,
            requests_at_current_stage=state.requests_at_current_stage,
            status=RolloutStatus.COMPLETED,
        )
    return CanaryRolloutState(
        stage_index=state.stage_index + 1,
        requests_at_current_stage=0,
        status=RolloutStatus.IN_PROGRESS,
    )


def mark_rolled_back(state: CanaryRolloutState) -> CanaryRolloutState:
    return CanaryRolloutState(
        stage_index=state.stage_index,
        requests_at_current_stage=state.requests_at_current_stage,
        status=RolloutStatus.ROLLED_BACK,
    )


__all__ = [
    "Arm",
    "CanaryConfig",
    "CanaryRolloutState",
    "RolloutStatus",
    "mark_rolled_back",
    "record_canary_request",
    "route_canary_traffic",
    "start_rollout",
    "try_advance_stage",
]
