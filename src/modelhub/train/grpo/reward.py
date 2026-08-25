"""B5's reward function — PLAN.md: "结果完全匹配 1.0 / 语法合法但结果错
0.0 / 语法错 0.0", reusing A2 (sqlexec) + A3 (compare) unchanged for the
execution+comparison that produces the `PredictionRecord` this module
scores.

★ This round's real difficulty (PLAN.md's own framing): reward-quality
protection, not the reward mapping itself. CLAUDE.md §2.3's single
biggest warning — "把系统错当模型错是全项目最容易犯、最难发现、后果
最严重的 bug" — applies to GRPO training more sharply than anywhere
else in this project: a masked-vs-zero mistake here doesn't just misgrade
one sample, it silently teaches the policy on pure noise while the loss
curve looks completely normal.

Masked (reward=`None`, never `0.0`, never enters an advantage
computation — a caller building a `datasets.Dataset`/loss mask for veRL
must filter these out, not zero-fill them):
  - `HARNESS_DB_UNAVAILABLE`/`HARNESS_INTERNAL` — PLAN.md ★ item 1.
  - `OUTPUT_TRUNCATED` — PLAN.md ★ item 2.
  - `ComparisonResult.UNDECIDABLE` (gold SQL itself failed to execute) —
    not named in PLAN.md's original wording, but the exact same
    "punishing the model for something that is not its fault" failure
    mode: if the gold result doesn't exist, there is no ground truth to
    score the model's SQL against, correct or not.
  - `UNCLASSIFIED` (`ErrorCode.UNCLASSIFIED`, CLAUDE.md §2.2's one
    permitted catch-all) — fault attribution is genuinely unknown here,
    so scoring it 0.0 would risk exactly the §2.3 hazard this module's
    docstring calls its central risk (silently teaching the policy on
    noise). Masked like the others, but tracked separately via its own
    `MaskReason.UNCLASSIFIED` and a rate threshold in
    `step_diagnostics.py` — never conflated with a confirmed harness
    fault, and never left to flood silently.

Scored (reward=1.0 or 0.0, the model's own outcome):
  - `EXEC_OK` + `EQUAL` -> 1.0
  - `EXEC_OK` + not `EQUAL` -> 0.0 (语法合法但结果错)
  - `SYNTAX`/`SEMANTIC`/`TIMEOUT` -> 0.0 (语法错, PLAN.md's literal
    wording covers all three model-fault SQL outcome codes)
  - `UNSAFE_STATEMENT` -> 0.0 (not named in PLAN.md; unambiguously the
    model's own fault, same reasoning B4's `dpo.py` already applied to
    the identical gap in its own 3-tier scheme)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.records import PredictionRecord

REWARD_FUNCTION_VERSION = "1.0.0"  # PLAN.md item 8: goes into RunManifest.reward_fn_version

_MODEL_FAULT_ZERO_CODES = frozenset(
    {ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT, ErrorCode.UNSAFE_STATEMENT}
)


class RewardOutcome(StrEnum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    MASKED = "MASKED"


class MaskReason(StrEnum):
    HARNESS_ERROR = "HARNESS_ERROR"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    UNDECIDABLE = "UNDECIDABLE"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass(frozen=True)
class RolloutReward:
    sample_id: str
    reward: float | None
    outcome: RewardOutcome
    mask_reason: MaskReason | None

    def __post_init__(self) -> None:
        if self.outcome is RewardOutcome.MASKED:
            if self.reward is not None or self.mask_reason is None:
                raise ValueError(
                    f"{self.sample_id}: MASKED outcome must carry reward=None and a "
                    f"mask_reason, got reward={self.reward!r} mask_reason={self.mask_reason!r}"
                )
        else:
            if self.reward is None or self.mask_reason is not None:
                raise ValueError(
                    f"{self.sample_id}: {self.outcome.value} outcome must carry a real "
                    f"reward and no mask_reason, got reward={self.reward!r} "
                    f"mask_reason={self.mask_reason!r}"
                )


def compute_reward(prediction: PredictionRecord) -> RolloutReward:
    """Whitelist-ordered, exhaustive over every real `PredictionRecord.
    exec_code`/`comparison_result` combination — an unhandled `exec_code`
    hard-fails rather than silently defaulting to a reward (CLAUDE.md
    §1.3)."""
    if prediction.is_harness_error:
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=None,
            outcome=RewardOutcome.MASKED,
            mask_reason=MaskReason.HARNESS_ERROR,
        )
    if prediction.is_output_truncated:
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=None,
            outcome=RewardOutcome.MASKED,
            mask_reason=MaskReason.OUTPUT_TRUNCATED,
        )
    if prediction.comparison_result is ComparisonResult.UNDECIDABLE:
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=None,
            outcome=RewardOutcome.MASKED,
            mask_reason=MaskReason.UNDECIDABLE,
        )
    if prediction.exec_code is ErrorCode.UNCLASSIFIED:
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=None,
            outcome=RewardOutcome.MASKED,
            mask_reason=MaskReason.UNCLASSIFIED,
        )
    if prediction.exec_code is ErrorCode.EXEC_OK:
        if prediction.comparison_result is ComparisonResult.EQUAL:
            return RolloutReward(
                sample_id=prediction.sample_id,
                reward=1.0,
                outcome=RewardOutcome.CORRECT,
                mask_reason=None,
            )
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=0.0,
            outcome=RewardOutcome.INCORRECT,
            mask_reason=None,
        )
    if prediction.exec_code in _MODEL_FAULT_ZERO_CODES:
        return RolloutReward(
            sample_id=prediction.sample_id,
            reward=0.0,
            outcome=RewardOutcome.INCORRECT,
            mask_reason=None,
        )
    raise ValueError(
        f"unhandled exec_code {prediction.exec_code!r} for sample {prediction.sample_id!r} — "
        f"compute_reward must be exhaustive, not silently default a reward (CLAUDE.md §1.3)"
    )


__all__ = [
    "REWARD_FUNCTION_VERSION",
    "MaskReason",
    "RewardOutcome",
    "RolloutReward",
    "compute_reward",
]
