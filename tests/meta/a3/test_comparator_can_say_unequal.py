"""CLAUDE.md §1.5 required meta-test: `test_comparator_can_say_unequal`.

A comparator that always returns EQUAL regardless of input is worse than
no comparator — it would make every gate, eval report, and GRPO reward
signal a rubber stamp. This injects a deterministically-unequal pair and
asserts NOT_EQUAL actually comes back, plus the complementary proofs that
UNDECIDABLE fires on error/truncation (CLAUDE.md/A3's other two "never
return EQUAL" guarantees) and that a genuinely-equal pair does return
EQUAL (so this file isn't just "always red" either).
"""

from modelhub.compare.comparator import compare_result_sets
from modelhub.compare.config import ComparatorConfig
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.sqlexec.outcomes import ResultSet


def _rs(rows: list[tuple], *, truncated: bool = False) -> ResultSet:
    return ResultSet(
        columns=["v"],
        rows=rows,
        truncated=truncated,
        row_limit=1000,
        rows_scanned=None if truncated else len(rows),
    )


def test_comparator_can_say_unequal() -> None:
    predicted = _rs([(1,)])
    gold = _rs([(2,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_comparator_can_say_equal_this_file_is_not_always_red() -> None:
    same = _rs([(1,)])
    outcome = compare_result_sets(same, same)
    assert outcome.result is ComparisonResult.EQUAL


def test_comparator_never_says_equal_when_truncated() -> None:
    predicted = _rs([(1,)], truncated=True)
    gold = _rs([(1,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.UNDECIDABLE
    assert outcome.reason is UndecidableReason.TOO_LARGE


def test_comparator_never_says_equal_on_internal_error() -> None:
    unhashable = _rs([(["not", "hashable"],)])  # type: ignore[list-item]
    outcome = compare_result_sets(
        unhashable, unhashable, config=ComparatorConfig(row_order="unordered")
    )
    assert outcome.result is ComparisonResult.UNDECIDABLE
    assert outcome.reason is UndecidableReason.INTERNAL_ERROR
