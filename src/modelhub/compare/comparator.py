"""The core three-state comparator.

Ordered comparison does an exact pairwise `_values_equal` check per cell —
no approximation. Unordered (the common case: most gold SQL has no
ORDER BY) canonicalizes each row into a hashable tuple and compares as a
multiset/set; canonicalizing a float means rounding it onto a grid sized
by `float_abs_tol`, which is an approximation of true tolerance-aware
multiset matching (the exact version is a bipartite-matching problem) —
documented here rather than silently assumed. It is still strictly more
correct than BIRD's own official scorer, which does zero float tolerance
at all (see official_baselines.py).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from modelhub.compare.config import ComparatorConfig
from modelhub.compare.result_types import ComparisonOutcome, UndecidableReason
from modelhub.sqlexec.outcomes import ResultSet

_NULL_TOKEN = "\x00NULL\x00"


def resolve_row_order(gold_sql: str) -> str:
    """Heuristic: gold SQL has an ORDER BY => rows are compared in order.

    Text-substring based, not a real SQL parser — same class of
    documented approximation as data/spider_hardness.py. A false positive
    (ORDER BY appears inside a subquery whose ordering doesn't survive to
    the outer result, or inside a string literal) would make the
    comparator stricter than necessary, not looser — it can only ever
    convert a would-be EQUAL into a false NOT_EQUAL, never the reverse, so
    it fails toward "flag for human review", not toward silently passing.
    """
    return "ordered" if "order by" in gold_sql.lower() else "unordered"


def _is_numeric(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _coerce_numeric(v: Any) -> float | None:
    if _is_numeric(v):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _values_equal(a: Any, b: Any, config: ComparatorConfig) -> bool:
    if a is None or b is None:
        if a is None and b is None:
            return config.null_equals_null
        return False

    if config.type_mode == "strict":
        # "strict" governs CROSS-type coercion only (1 vs 1.0 vs "1") —
        # float-vs-float tolerance (dimension 4) still always applies
        # within the same type, it is an orthogonal dimension.
        if type(a) is not type(b):
            return False
        if isinstance(a, float):
            return math.isclose(a, b, rel_tol=config.float_rel_tol, abs_tol=config.float_abs_tol)
        return bool(a == b)

    # coerce_numeric_and_string
    if _is_numeric(a) and _is_numeric(b):
        return math.isclose(
            float(a), float(b), rel_tol=config.float_rel_tol, abs_tol=config.float_abs_tol
        )
    na, nb = _coerce_numeric(a), _coerce_numeric(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=config.float_rel_tol, abs_tol=config.float_abs_tol)
    if type(a) is not type(b):
        return False
    return bool(a == b)


def _canonicalize_value(v: Any, config: ComparatorConfig) -> Any:
    if v is None:
        return (
            _NULL_TOKEN if config.null_equals_null else object()
        )  # never hash-equal to itself twice

    if config.type_mode == "coerce_numeric_and_string":
        n = _coerce_numeric(v)
        if n is not None:
            v = n
        if isinstance(v, float):
            return _canonicalize_float(v, config)
        return v

    # strict: Python's own `==`/hash treat 1 == 1.0 (and hash equal) —
    # tag with the concrete type so canonicalization doesn't silently
    # coerce across types the way Python's own equality does. Tolerance
    # still applies float-vs-float (dimension 4 stays orthogonal to
    # dimension 7 — see _values_equal's docstring-equivalent comment).
    if isinstance(v, float):
        return ("float", _canonicalize_float(v, config))
    return (type(v).__name__, v)


def _canonicalize_float(v: float, config: ComparatorConfig) -> float:
    grid = round(v / config.float_abs_tol) * config.float_abs_tol if config.float_abs_tol > 0 else v
    return round(grid, 12)


def _canonicalize_row(row: tuple[Any, ...], config: ComparatorConfig) -> tuple[Any, ...]:
    return tuple(_canonicalize_value(v, config) for v in row)


def _reorder_by_name(result: ResultSet, column_order: list[str]) -> list[tuple[Any, ...]]:
    index_of = {name: i for i, name in enumerate(result.columns)}
    return [tuple(row[index_of[name]] for name in column_order) for row in result.rows]


_DIFF_EXAMPLE_LIMIT = 10


def _example_rows(rows: list[Any]) -> dict[str, Any]:
    """Up to `_DIFF_EXAMPLE_LIMIT` example differing rows for a NOT_EQUAL
    outcome's debug context — NOT the comparison result itself, which is
    always computed over the full, untruncated row set before this is
    called. The cap is explicit in the returned data (mirrors
    ResultSet.truncated/row_limit), not a silent slice the caller has to
    already know about.
    """
    # check-no-cheating: allow=SILENT_SLICE_TRUNCATION reason=debug-only, verdict already final
    examples = rows[:_DIFF_EXAMPLE_LIMIT]
    return {
        "examples": examples,
        "example_limit": _DIFF_EXAMPLE_LIMIT,
        "total_differing_rows": len(rows),
        "truncated": len(rows) > _DIFF_EXAMPLE_LIMIT,
    }


def compare_result_sets(
    predicted: ResultSet, gold: ResultSet, *, config: ComparatorConfig | None = None
) -> ComparisonOutcome:
    """Compare two SQL result sets. Never returns EQUAL on error or on a
    truncated input — see module/result_types docstrings."""
    config = config or ComparatorConfig()
    try:
        return _compare_result_sets_inner(predicted, gold, config)
    except Exception as e:  # comparator bugs must resolve to UNDECIDABLE, never EQUAL
        return ComparisonOutcome.undecidable(
            UndecidableReason.INTERNAL_ERROR,
            detail=f"{type(e).__name__}: {e}",
        )


def _reconcile_columns(
    predicted: ResultSet, gold: ResultSet, config: ComparatorConfig
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]] | ComparisonOutcome:
    """Returns the (possibly reordered) row lists to compare, or a
    NOT_EQUAL outcome if the column shapes can't be reconciled at all."""
    if config.column_match_mode == "by_position":
        if len(predicted.columns) != len(gold.columns):
            return ComparisonOutcome.not_equal(
                detail="column count differs",
                context={
                    "predicted_columns": len(predicted.columns),
                    "gold_columns": len(gold.columns),
                },
            )
        return predicted.rows, gold.rows

    if set(predicted.columns) != set(gold.columns):
        return ComparisonOutcome.not_equal(
            detail="column name sets differ",
            context={"predicted_columns": predicted.columns, "gold_columns": gold.columns},
        )
    column_order = gold.columns
    return _reorder_by_name(predicted, column_order), _reorder_by_name(gold, column_order)


def _compare_ordered(
    predicted_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    config: ComparatorConfig,
) -> ComparisonOutcome:
    if len(predicted_rows) != len(gold_rows):
        return ComparisonOutcome.not_equal(
            detail="row count differs",
            context={"predicted_rows": len(predicted_rows), "gold_rows": len(gold_rows)},
        )
    for i, (p_row, g_row) in enumerate(zip(predicted_rows, gold_rows, strict=True)):
        if len(p_row) != len(g_row):
            return ComparisonOutcome.not_equal(detail=f"row {i}: arity mismatch")
        if not all(_values_equal(p, g, config) for p, g in zip(p_row, g_row, strict=True)):
            return ComparisonOutcome.not_equal(
                detail=f"row {i} differs in ordered comparison",
                context={"row_index": i, "predicted": p_row, "gold": g_row},
            )
    return ComparisonOutcome.equal(detail="ordered comparison matched")


def _compare_unordered(
    predicted_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    config: ComparatorConfig,
) -> ComparisonOutcome:
    predicted_canon = [_canonicalize_row(r, config) for r in predicted_rows]
    gold_canon = [_canonicalize_row(r, config) for r in gold_rows]

    if config.duplicate_row_mode == "set":
        if set(predicted_canon) == set(gold_canon):
            return ComparisonOutcome.equal(detail="unordered set comparison matched")
        return ComparisonOutcome.not_equal(
            detail="unordered set comparison differs",
            context={"predicted_only": _example_rows(list(set(predicted_canon) - set(gold_canon)))},
        )

    predicted_counter = Counter(predicted_canon)
    gold_counter = Counter(gold_canon)
    if predicted_counter == gold_counter:
        return ComparisonOutcome.equal(detail="unordered multiset comparison matched")
    diff = predicted_counter - gold_counter
    return ComparisonOutcome.not_equal(
        detail="unordered multiset comparison differs (row present a different number of times, "
        "or not present at all)",
        context={"predicted_extra_rows": _example_rows(list(diff.elements()))},
    )


def _compare_result_sets_inner(
    predicted: ResultSet, gold: ResultSet, config: ComparatorConfig
) -> ComparisonOutcome:
    if predicted.truncated or gold.truncated:
        return ComparisonOutcome.undecidable(
            UndecidableReason.TOO_LARGE,
            detail="one or both result sets were truncated at their row_limit",
            context={"predicted_truncated": predicted.truncated, "gold_truncated": gold.truncated},
        )

    if config.both_empty_is_equal and not predicted.rows and not gold.rows:
        return ComparisonOutcome.equal(detail="both result sets are empty")

    reconciled = _reconcile_columns(predicted, gold, config)
    if isinstance(reconciled, ComparisonOutcome):
        return reconciled
    predicted_rows, gold_rows = reconciled

    if config.row_order == "ordered":
        return _compare_ordered(predicted_rows, gold_rows, config)
    return _compare_unordered(predicted_rows, gold_rows, config)
