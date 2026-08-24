"""The seven comparison dimensions — every one an explicit config field.

CLAUDE.md/A3: column order, column-name vs positional matching, row order,
NULL semantics, float tolerance, empty-result handling, duplicate-row
semantics (multiset vs set), and type coercion must each be a named,
versioned config value, not an implicit assumption buried in comparison
logic. `ComparatorConfig` is itself `extra="forbid"`, same as every other
config in this project (CLAUDE.md §4) — a typo'd dimension name must
error, not silently be ignored.
"""

from __future__ import annotations

from typing import Literal

from modelhub.common.config import ModelHubBaseConfig

COMPARATOR_VERSION = "1.0.0"


class ComparatorConfig(ModelHubBaseConfig):
    # 1. column order: compare column-by-column at the same tuple index
    # ("by_position", robust to SELECT list aliasing differences) or
    # require each named column to independently line up ("by_name").
    column_match_mode: Literal["by_position", "by_name"] = "by_position"

    # 2. row order: "ordered" requires rows in the same sequence;
    # "unordered" compares as (multi)sets regardless of sequence. Per
    # CLAUDE.md §A3, callers should pass "ordered" only when the gold SQL
    # itself has an ORDER BY — an unordered gold query has no defined row
    # order to hold the prediction to.
    row_order: Literal["ordered", "unordered"] = "unordered"

    # 3. NULL semantics: SQL's own NULL != NULL is the wrong default for
    # comparing two *result sets* (as opposed to evaluating a SQL
    # predicate) — two rows that both have NULL in the same slot are the
    # same result. Kept configurable since some report generators may want
    # strict SQL semantics for auditing purposes.
    null_equals_null: bool = True

    # 4. float tolerance: never bare `==` on floats.
    float_rel_tol: float = 1e-6
    float_abs_tol: float = 1e-9

    # 5. empty result handling: both sides empty is EQUAL by default (it's
    # the trivially correct comparison); set False to route all-empty
    # comparisons to UNDECIDABLE instead, for pipelines that consider an
    # empty result suspicious rather than possibly-correct.
    both_empty_is_equal: bool = True

    # 6. duplicate rows: "multiset" (correct SQL semantics: row counts
    # matter) vs "set" (collapses duplicates — this is what BIRD's own
    # official scorer does; see compare/official_baselines.py). Exposed so
    # the consistency-rate script can reproduce the official baseline's
    # behavior exactly, not just approximate it.
    duplicate_row_mode: Literal["multiset", "set"] = "multiset"

    # 7. type coercion: "strict" means 1 (int) != 1.0 (float) != "1" (str);
    # "coerce_numeric_and_string" normalizes numeric-looking values across
    # int/float/numeric-string before comparing.
    type_mode: Literal["strict", "coerce_numeric_and_string"] = "coerce_numeric_and_string"
