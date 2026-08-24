"""CLAUDE.md §1.5 required meta-test: proves B4's preference-pair
construction actually enforces the two hard rules PLAN.md states (★
system-error samples never enter a pair; 同级内不配对 — same-tier pairs
never form) by constructing inputs specifically designed to violate each
rule if the implementation were wrong, and proves the mechanism is not
permanently red (a normal, healthy mixed-tier input DOES produce pairs).
"""

from __future__ import annotations

from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.eval.records import PredictionRecord
from modelhub.train.dpo import DiscardReason, build_preference_pairs_for_question


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


def test_harness_error_candidate_never_enters_any_pair_even_as_the_only_alternative() -> None:
    """If pairing logic were broken (e.g. treating "discarded" as just
    another tier), a harness-error sample sitting alongside a correct
    one would still get paired — this asserts that never happens."""
    predictions = [
        _prediction(predicted_sql="correct"),
        _prediction(
            predicted_sql="harness_fail",
            exec_code=ErrorCode.HARNESS_INTERNAL,
            comparison_result=None,
        ),
    ]
    result = build_preference_pairs_for_question("q0", predictions)
    for pair in result.pairs:
        assert pair.chosen_sql != "harness_fail"
        assert pair.rejected_sql != "harness_fail"
    assert result.pairs == ()  # only one real kept candidate — nothing to pair


def test_undecidable_and_truncated_never_enter_a_pair() -> None:
    predictions = [
        _prediction(predicted_sql="correct"),
        _prediction(
            predicted_sql="undecidable",
            comparison_result=ComparisonResult.UNDECIDABLE,
            undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
        ),
        _prediction(
            predicted_sql="truncated", finish_reason="length", exec_code=ErrorCode.OUTPUT_TRUNCATED
        ),
    ]
    result = build_preference_pairs_for_question("q0", predictions)
    discarded_sql = {"undecidable", "truncated"}
    for pair in result.pairs:
        assert pair.chosen_sql not in discarded_sql
        assert pair.rejected_sql not in discarded_sql
    discarded = [c for c in result.classified if c.discard_reason is not None]
    assert {c.discard_reason for c in discarded} == {
        DiscardReason.UNDECIDABLE,
        DiscardReason.OUTPUT_TRUNCATED,
    }


def test_four_identical_tier_candidates_produce_zero_pairs() -> None:
    """同级内不配对 — this would fail loudly (produce spurious pairs) if
    the implementation paired same-tier candidates."""
    predictions = [_prediction(predicted_sql=f"correct{i}") for i in range(4)]
    result = build_preference_pairs_for_question("q0", predictions)
    assert result.pairs == ()


def test_a_healthy_mixed_tier_question_is_not_permanently_pairless() -> None:
    """Proves the mechanism isn't stuck at zero pairs: a real mixed-tier
    k=4 sample produces real cross-tier pairs."""
    predictions = [
        _prediction(predicted_sql="correct"),
        _prediction(predicted_sql="wrong", comparison_result=ComparisonResult.NOT_EQUAL),
        _prediction(predicted_sql="bad_sql", exec_code=ErrorCode.SYNTAX, comparison_result=None),
        _prediction(predicted_sql="correct2"),
    ]
    result = build_preference_pairs_for_question("q0", predictions)
    assert len(result.pairs) > 0
