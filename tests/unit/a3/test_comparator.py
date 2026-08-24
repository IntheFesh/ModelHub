from modelhub.compare.comparator import compare_result_sets, resolve_row_order
from modelhub.compare.config import ComparatorConfig
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.sqlexec.outcomes import ResultSet


def _rs(
    columns: list[str], rows: list[tuple], *, truncated: bool = False, row_limit: int = 1000
) -> ResultSet:
    return ResultSet(
        columns=columns,
        rows=rows,
        truncated=truncated,
        row_limit=row_limit,
        rows_scanned=None if truncated else len(rows),
    )


# ── basics ───────────────────────────────────────────────────────────────


def test_identical_results_are_equal() -> None:
    a = _rs(["id"], [(1,), (2,)])
    outcome = compare_result_sets(a, a)
    assert outcome.result is ComparisonResult.EQUAL


def test_different_results_are_not_equal() -> None:
    a = _rs(["id"], [(1,)])
    b = _rs(["id"], [(2,)])
    outcome = compare_result_sets(a, b)
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_truncated_side_is_undecidable_never_equal() -> None:
    a = _rs(["id"], [(1,)] * 5, truncated=True, row_limit=5)
    b = _rs(["id"], [(1,)] * 5)
    outcome = compare_result_sets(a, b)
    assert outcome.result is ComparisonResult.UNDECIDABLE
    assert outcome.reason is UndecidableReason.TOO_LARGE


def test_internal_error_never_becomes_equal() -> None:
    # An unhashable cell value (e.g. a driver returning a list/dict for a
    # JSON column) breaks the unordered path's canonicalize-then-hash
    # comparison. Whatever goes wrong inside the comparator, the outer
    # try/except in compare_result_sets must turn it into UNDECIDABLE, not
    # let it crash the caller or — worse — silently resolve to EQUAL.
    a = _rs(["id", "tags"], [(1, ["a", "b"])])  # type: ignore[list-item]
    b = _rs(["id", "tags"], [(1, ["a", "b"])])  # type: ignore[list-item]
    outcome = compare_result_sets(a, b, config=ComparatorConfig(row_order="unordered"))
    assert outcome.result is ComparisonResult.UNDECIDABLE
    assert outcome.reason is UndecidableReason.INTERNAL_ERROR


# ── dimension 1: column order / aliasing ────────────────────────────────


def test_column_order_by_position_cares_about_order() -> None:
    predicted = _rs(["name", "id"], [("alice", 1)])
    gold = _rs(["id", "name"], [(1, "alice")])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(column_match_mode="by_position")
    )
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_column_order_by_name_ignores_order() -> None:
    predicted = _rs(["name", "id"], [("alice", 1)])
    gold = _rs(["id", "name"], [(1, "alice")])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(column_match_mode="by_name")
    )
    assert outcome.result is ComparisonResult.EQUAL


def test_column_name_mismatch_by_name_is_not_equal() -> None:
    predicted = _rs(["cnt"], [(5,)])
    gold = _rs(["total"], [(5,)])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(column_match_mode="by_name")
    )
    assert outcome.result is ComparisonResult.NOT_EQUAL


# ── dimension 2: row order ───────────────────────────────────────────────


def test_resolve_row_order_detects_order_by() -> None:
    assert resolve_row_order("SELECT * FROM t ORDER BY id") == "ordered"
    assert resolve_row_order("SELECT * FROM t") == "unordered"


def test_ordered_mode_cares_about_sequence() -> None:
    predicted = _rs(["id"], [(2,), (1,)])
    gold = _rs(["id"], [(1,), (2,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(row_order="ordered"))
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_unordered_mode_ignores_sequence() -> None:
    predicted = _rs(["id"], [(2,), (1,)])
    gold = _rs(["id"], [(1,), (2,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(row_order="unordered"))
    assert outcome.result is ComparisonResult.EQUAL


# ── dimension 3: NULL semantics ──────────────────────────────────────────


def test_null_equals_null_true_by_default() -> None:
    predicted = _rs(["v"], [(None,)])
    gold = _rs(["v"], [(None,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.EQUAL


def test_null_equals_null_false_makes_null_never_match() -> None:
    predicted = _rs(["v"], [(None,)])
    gold = _rs(["v"], [(None,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(null_equals_null=False))
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_null_never_equals_a_real_value_regardless_of_config() -> None:
    predicted = _rs(["v"], [(None,)])
    gold = _rs(["v"], [(0,)])
    for null_equals_null in (True, False):
        outcome = compare_result_sets(
            predicted, gold, config=ComparatorConfig(null_equals_null=null_equals_null)
        )
        assert outcome.result is ComparisonResult.NOT_EQUAL


# ── dimension 4: float tolerance ─────────────────────────────────────────


def test_float_within_tolerance_is_equal() -> None:
    predicted = _rs(["v"], [(3.0000000001,)])
    gold = _rs(["v"], [(3.0,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.EQUAL


def test_float_beyond_tolerance_is_not_equal() -> None:
    predicted = _rs(["v"], [(3.1,)])
    gold = _rs(["v"], [(3.0,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_float_tolerance_is_never_a_bare_equality_check() -> None:
    # Regression guard: a naive `==` would fail this even though it's the
    # canonical "these floats are the same number" case.
    predicted = _rs(["v"], [(0.1 + 0.2,)])  # 0.30000000000000004 in IEEE 754
    gold = _rs(["v"], [(0.3,)])
    assert (0.1 + 0.2) != 0.3  # sanity: bare == really would fail here
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.EQUAL


# ── dimension 5: empty result ─────────────────────────────────────────────


def test_both_empty_is_equal_by_default() -> None:
    predicted = _rs(["id"], [])
    gold = _rs(["id"], [])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.EQUAL


def test_both_empty_is_equal_can_be_disabled() -> None:
    predicted = _rs(["id"], [])
    gold = _rs(["id"], [])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(both_empty_is_equal=False)
    )
    # multiset comparison of two empty lists is still trivially equal —
    # disabling the fast-path doesn't change the answer, only the code path.
    assert outcome.result is ComparisonResult.EQUAL


def test_one_empty_one_not_is_not_equal() -> None:
    predicted = _rs(["id"], [])
    gold = _rs(["id"], [(1,)])
    outcome = compare_result_sets(predicted, gold)
    assert outcome.result is ComparisonResult.NOT_EQUAL


# ── dimension 6: duplicate rows (multiset vs set) ────────────────────────


def test_multiset_mode_cares_about_duplicate_counts() -> None:
    predicted = _rs(["id"], [(1,), (1,), (2,)])
    gold = _rs(["id"], [(1,), (2,)])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(duplicate_row_mode="multiset")
    )
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_set_mode_ignores_duplicate_counts() -> None:
    predicted = _rs(["id"], [(1,), (1,), (2,)])
    gold = _rs(["id"], [(1,), (2,)])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(duplicate_row_mode="set")
    )
    assert outcome.result is ComparisonResult.EQUAL


# ── dimension 7: type coercion ────────────────────────────────────────────


def test_strict_type_mode_distinguishes_int_float_string() -> None:
    predicted = _rs(["v"], [(1,)])
    gold = _rs(["v"], [(1.0,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(type_mode="strict"))
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_strict_type_mode_distinguishes_int_float_in_ordered_comparison_too() -> None:
    # Regression test: an earlier version's `_values_equal` (the ordered-
    # comparison path) ran the numeric-tolerance branch unconditionally
    # before checking type_mode, so "strict" only worked by accident for
    # the unordered/canonicalize path and not this one.
    predicted = _rs(["v"], [(1,)])
    gold = _rs(["v"], [(1.0,)])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(type_mode="strict", row_order="ordered")
    )
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_strict_type_mode_distinguishes_bool_from_int() -> None:
    # Python's own `True == 1` — strict mode must not inherit that.
    predicted = _rs(["v"], [(True,)])
    gold = _rs(["v"], [(1,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(type_mode="strict"))
    assert outcome.result is ComparisonResult.NOT_EQUAL


def test_strict_type_mode_still_applies_float_tolerance_within_same_type() -> None:
    # "strict" only turns off cross-type coercion — float-vs-float
    # tolerance (dimension 4) is orthogonal and must still apply.
    predicted = _rs(["v"], [(3.0000000001,)])
    gold = _rs(["v"], [(3.0,)])
    outcome = compare_result_sets(predicted, gold, config=ComparatorConfig(type_mode="strict"))
    assert outcome.result is ComparisonResult.EQUAL


def test_coerce_mode_treats_int_float_string_as_equal() -> None:
    for predicted_val in (1, 1.0, "1"):
        predicted = _rs(["v"], [(predicted_val,)])
        gold = _rs(["v"], [(1,)])
        outcome = compare_result_sets(
            predicted, gold, config=ComparatorConfig(type_mode="coerce_numeric_and_string")
        )
        assert outcome.result is ComparisonResult.EQUAL, f"failed for {predicted_val!r}"


def test_non_numeric_string_never_coerces() -> None:
    predicted = _rs(["v"], [("abc",)])
    gold = _rs(["v"], [(1,)])
    outcome = compare_result_sets(
        predicted, gold, config=ComparatorConfig(type_mode="coerce_numeric_and_string")
    )
    assert outcome.result is ComparisonResult.NOT_EQUAL
