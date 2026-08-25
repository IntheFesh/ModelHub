"""CLAUDE.md §1.5 required meta-test: proves B5's most important safety
property — system errors are masked, never scored as a model failure —
actually holds by injecting every masking scenario and asserting the
real reward is `None` with a real `MaskReason`, not a silent `0.0`. Also
proves the degenerate-group and harness-flood-abort mechanisms genuinely
fire, and none of them is permanently red."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.eval.records import PredictionRecord
from modelhub.train.grpo.group_diagnostics import RolloutGroup, diagnose_group
from modelhub.train.grpo.reward import RewardOutcome, compute_reward
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


@pytest.mark.parametrize(
    "prediction_overrides",
    [
        {"exec_code": ErrorCode.HARNESS_DB_UNAVAILABLE, "comparison_result": None},
        {"exec_code": ErrorCode.HARNESS_INTERNAL, "comparison_result": None},
        {
            "finish_reason": "length",
            "exec_code": ErrorCode.OUTPUT_TRUNCATED,
            "comparison_result": None,
        },
        {
            "comparison_result": ComparisonResult.UNDECIDABLE,
            "undecidable_reason": UndecidableReason.GOLD_EXEC_FAILED,
        },
        {"exec_code": ErrorCode.UNCLASSIFIED, "comparison_result": None},
    ],
)
def test_every_masking_scenario_produces_none_reward_not_zero(
    prediction_overrides: dict[str, object],
) -> None:
    """The exact bug CLAUDE.md §2.3 names as the project's most
    dangerous: a system fault silently scored as `0.0` reads
    indistinguishably from a real model failure to a GRPO trainer."""
    reward = compute_reward(_prediction(**prediction_overrides))
    assert reward.outcome is RewardOutcome.MASKED
    assert reward.reward is None  # not 0.0 — CLAUDE.md §2.4: missing is None
    assert reward.mask_reason is not None


def test_a_real_model_failure_still_gets_a_real_zero() -> None:
    """Proves masking isn't stuck on: an ordinary syntax error (the
    model's own fault) is scored 0.0, not masked."""
    reward = compute_reward(_prediction(exec_code=ErrorCode.SYNTAX, comparison_result=None))
    assert reward.outcome is RewardOutcome.INCORRECT
    assert reward.reward == 0.0


def test_degenerate_group_detection_fires_on_all_same_reward() -> None:
    from modelhub.train.grpo.reward import RolloutReward

    group = RolloutGroup(
        question_id="q0",
        rewards=tuple(
            RolloutReward(
                sample_id="q0", reward=0.0, outcome=RewardOutcome.INCORRECT, mask_reason=None
            )
            for _ in range(4)
        ),
    )
    assert diagnose_group(group).is_degenerate is True


def test_degenerate_group_detection_does_not_fire_on_healthy_variance() -> None:
    from modelhub.train.grpo.reward import RolloutReward

    group = RolloutGroup(
        question_id="q0",
        rewards=(
            RolloutReward(
                sample_id="q0", reward=1.0, outcome=RewardOutcome.CORRECT, mask_reason=None
            ),
            RolloutReward(
                sample_id="q0", reward=0.0, outcome=RewardOutcome.INCORRECT, mask_reason=None
            ),
        ),
    )
    assert diagnose_group(group).is_degenerate is False


def test_harness_error_flood_genuinely_aborts_training() -> None:
    from modelhub.train.grpo.group_diagnostics import GroupDiagnostics

    predictions = [_prediction()] * 90 + [
        _prediction(exec_code=ErrorCode.HARNESS_INTERNAL, comparison_result=None)
    ] * 10  # 10% harness-error rate, well above the 1% threshold
    diag = compute_step_diagnostics(
        1,
        predictions,
        [
            GroupDiagnostics(
                question_id="q0",
                total_rollouts=1,
                masked_count=0,
                scored_rewards=(1.0,),
                is_degenerate=False,
                degenerate_reason=None,
            )
        ],
    )
    with pytest.raises(ValueError, match="aborting GRPO training"):
        assert_harness_error_rate_ok(diag)


def test_healthy_harness_error_rate_does_not_abort() -> None:
    """Proves the abort check is not permanently red: a clean run
    (0% harness errors) passes through without raising."""
    from modelhub.train.grpo.group_diagnostics import GroupDiagnostics

    predictions = [_prediction()] * 100
    diag = compute_step_diagnostics(
        1,
        predictions,
        [
            GroupDiagnostics(
                question_id="q0",
                total_rollouts=1,
                masked_count=0,
                scored_rewards=(1.0,),
                is_degenerate=False,
                degenerate_reason=None,
            )
        ],
    )
    assert_harness_error_rate_ok(diag)  # must not raise


def test_unclassified_flood_genuinely_aborts_training() -> None:
    # Regression test: a rising UNCLASSIFIED rate used to be invisible to
    # every system-error-rate check in this module (PredictionRecord.
    # is_harness_error only covers HARNESS_DB_UNAVAILABLE/HARNESS_INTERNAL,
    # so a 100% unclassified step could previously sail through
    # assert_harness_error_rate_ok reading a clean 0% harness_error_rate).
    from modelhub.train.grpo.group_diagnostics import GroupDiagnostics

    predictions = [_prediction()] * 90 + [
        _prediction(exec_code=ErrorCode.UNCLASSIFIED, comparison_result=None)
    ] * 10  # 10% unclassified rate, well above the 1% threshold
    diag = compute_step_diagnostics(
        1,
        predictions,
        [
            GroupDiagnostics(
                question_id="q0",
                total_rollouts=1,
                masked_count=0,
                scored_rewards=(1.0,),
                is_degenerate=False,
                degenerate_reason=None,
            )
        ],
    )
    assert_harness_error_rate_ok(diag)  # a pure unclassified flood is NOT a harness flood
    with pytest.raises(ValueError, match="aborting GRPO training"):
        assert_unclassified_rate_ok(diag)


def test_healthy_unclassified_rate_does_not_abort() -> None:
    """Proves the unclassified-rate abort check is not permanently red."""
    from modelhub.train.grpo.group_diagnostics import GroupDiagnostics

    predictions = [_prediction()] * 100
    diag = compute_step_diagnostics(
        1,
        predictions,
        [
            GroupDiagnostics(
                question_id="q0",
                total_rollouts=1,
                masked_count=0,
                scored_rewards=(1.0,),
                is_degenerate=False,
                degenerate_reason=None,
            )
        ],
    )
    assert_unclassified_rate_ok(diag)  # must not raise
