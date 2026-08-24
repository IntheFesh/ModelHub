"""Spider-style SQL complexity classification (easy/medium/hard/extra).

Spider's raw dataset files do NOT ship a difficulty field — the official
eval script (`taoyds/spider`, Apache-2.0) computes it from the *parsed*
SQL at eval time. This module has two layers of confidence, and they are
not the same:

1. **Threshold logic — verified, byte-exact.** We had network access to
   `raw.githubusercontent.com` (confirmed reachable; `huggingface.co` and
   `extensions.duckdb.org` are not — see docs/design-decisions.md DD-0003)
   and fetched the real script on 2026-08-24:
   https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
   `Evaluator.eval_hardness` there is exactly the four `if`/`elif` branches
   below, over three counts named `count_component1` / `count_component2`
   / `count_others`. Ported verbatim (only the variable names shortened),
   Apache-2.0, attribution: Yu, Tao et al., "Spider: A Large-Scale
   Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and
   Text-to-SQL Task", EMNLP 2018 / https://github.com/taoyds/spider.

2. **Component counting — still a text heuristic, not schema-aware.** The
   official script computes component1/component2/others by walking
   Spider's own parsed-SQL dict (built by `process_sql.py`, which needs
   the DB schema and NLTK's tokenizer — a materially bigger dependency
   chain we have not vendored here). `estimate_spider_hardness` instead
   regexes the raw SQL text, which is close but NOT identical:
     - now counts `LIMIT` (component1) — an earlier version of this
       function missed it entirely, caught when this file was updated
       against the real script.
     - `JOIN` is counted by the literal keyword; the official version
       counts `len(table_units) - 1`, so an old-style implicit join
       (`FROM a, b WHERE a.id = b.id`) is undercounted here.
     - `OR`/`LIKE` counts are whole-string keyword matches, not scoped to
       WHERE/HAVING/FROM condition tokens, so either could be thrown off
       by a string literal containing those words (rare in practice for
       Text2SQL gold SQL, not impossible).
   `FACTS.md`'s Spider dev distribution (easy 248/medium 446/hard 174/
   extra 166) is the official script's output; this module's output has
   NOT been diffed against it sample-by-sample. Do not cite this
   function's output as an official Spider hardness number in any report
   (CLAUDE.md §3.5) until that diff has actually been run.
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


def _count_component1(lower: str) -> int:
    count = 0
    count += 1 if " where " in f" {lower} " else 0
    count += 1 if "group by" in lower else 0
    count += 1 if "order by" in lower else 0
    count += 1 if re.search(r"\blimit\b", lower) else 0
    count += _count_keyword(lower, "join")
    count += _count_keyword(lower, "or")
    count += _count_keyword(lower, "like")
    return count


def _count_component2(lower: str) -> int:
    n_set_ops = sum(_count_keyword(lower, op) for op in _SET_OPS)
    n_nested = lower.count("select") - 1  # subqueries beyond the outermost SELECT
    return n_set_ops + max(0, n_nested)


def _count_others(lower: str) -> int:
    count = 0
    n_agg = sum(_count_keyword(lower, fn) for fn in _AGG_FUNCS)
    count += 1 if n_agg > 1 else 0
    count += 1 if _select_column_count(lower) > 1 else 0
    count += 1 if " where " in f" {lower} " and (" and " in lower or " or " in lower) else 0
    count += 1 if "group by" in lower and "having" in lower else 0
    return count


def estimate_spider_hardness(sql: str) -> SpiderHardness:
    """See module docstring: threshold logic is a verified port of the
    official `Evaluator.eval_hardness`; component counting is a text
    heuristic standing in for the official schema-aware SQL parser."""
    lower = sql.strip().lower()
    comp1 = _count_component1(lower)
    comp2 = _count_component2(lower)
    others = _count_others(lower)

    # -- verbatim port of taoyds/spider evaluation.py Evaluator.eval_hardness --
    if comp1 <= 1 and others == 0 and comp2 == 0:
        return SpiderHardness.EASY
    if comp2 == 0 and ((others <= 2 and comp1 <= 1) or (comp1 <= 2 and others < 2)):
        return SpiderHardness.MEDIUM
    if comp2 == 0 and ((others > 2 and comp1 <= 2) or (2 < comp1 <= 3 and others <= 2)):
        return SpiderHardness.HARD
    if comp1 <= 1 and others == 0 and comp2 <= 1:
        return SpiderHardness.HARD
    return SpiderHardness.EXTRA
