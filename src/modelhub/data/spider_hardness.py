"""Spider-style SQL complexity classification (easy/medium/hard/extra).

★ Honesty note (CLAUDE.md §12: "不确定就说不确定"): Spider's raw dataset
files do NOT ship a difficulty field — the official Spider evaluation
script (Yale-LILY's `evaluation.py`) computes hardness from the parsed SQL
at eval time, by counting query components (WHERE/GROUP BY/ORDER BY/JOIN,
set operations, nested subqueries, aggregations). We have no network
access in this sandbox to vendor that script and diff our output against
it sample-by-sample, so this is a **reimplementation from the documented
methodology, not a verified byte-exact port**. `FACTS.md`'s Spider dev
distribution numbers (easy 248/medium 446/hard 174/extra 166) come from
the official script; this module's output has NOT been validated against
them. Treat `estimate_spider_hardness` as a best-effort placeholder to be
checked against the real `evaluation.py` once a networked machine is
available — do not cite its output as an official Spider hardness number
in any report (CLAUDE.md §3.5).
"""

from __future__ import annotations

import re
from enum import StrEnum

_AGG_FUNCS = ("count", "sum", "avg", "min", "max")
_SET_OPS = ("union", "intersect", "except")


class SpiderHardness(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTRA = "extra"


def _count_keyword(sql_lower: str, keyword: str) -> int:
    return len(re.findall(rf"\b{re.escape(keyword)}\b", sql_lower))


def _select_column_count(sql_lower: str) -> int:
    match = re.search(r"select\s+(distinct\s+)?(.*?)\s+from\s", sql_lower, re.DOTALL)
    if not match:
        return 1
    cols = match.group(2)
    # split on top-level commas (naive but fine for a component-count heuristic,
    # not a real SQL parser)
    return max(1, cols.count(",") + 1)


def estimate_spider_hardness(sql: str) -> SpiderHardness:
    """Best-effort component-count classifier — see module docstring caveat."""
    lower = sql.strip().lower()

    n_where = 1 if " where " in f" {lower} " else 0
    n_group_by = 1 if "group by" in lower else 0
    n_order_by = 1 if "order by" in lower else 0
    n_join = _count_keyword(lower, "join")
    n_or = _count_keyword(lower, "or")
    n_like = _count_keyword(lower, "like")
    component1 = n_where + n_group_by + n_order_by + n_join + n_or + n_like

    n_set_ops = sum(_count_keyword(lower, op) for op in _SET_OPS)
    n_nested = lower.count("select") - 1  # subqueries beyond the outermost SELECT
    component2 = n_set_ops + max(0, n_nested)

    n_agg = sum(_count_keyword(lower, fn) for fn in _AGG_FUNCS)
    n_select_cols = _select_column_count(lower)
    others = 0
    others += 1 if n_agg > 1 else 0
    others += 1 if n_select_cols > 1 else 0
    others += 1 if n_where and lower.count(" and ") + lower.count(" or ") >= 1 else 0
    others += 1 if n_group_by and "having" in lower else 0

    if component1 <= 1 and others == 0 and component2 == 0:
        return SpiderHardness.EASY
    if component2 == 0 and ((others <= 2 and component1 <= 1) or (component1 <= 2 and others < 2)):
        return SpiderHardness.MEDIUM
    if component2 == 0 and (
        (others > 2 and component1 <= 2) or (2 < component1 <= 3 and others <= 2)
    ):
        return SpiderHardness.HARD
    return SpiderHardness.EXTRA
