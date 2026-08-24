"""A faithful port of BIRD's own official execution-accuracy comparator.

Fetched and read from the real source on 2026-08-24 (this sandbox can
reach `raw.githubusercontent.com`, confirmed reachable — see
docs/design-decisions.md DD-0009):
https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/bird/llm/src/evaluation.py
(MIT License, Alibaba Research / BIRD-bench). The comparison core is
exactly:

    if set(predicted_res) == set(ground_truth_res):
        res = 1

i.e. both result sets are dumped through Python's `set()` and compared.
This is the actual, real baseline our comparator (compare/comparator.py)
is meant to improve on, not a hypothetical — and diffing against it here
makes that improvement concrete rather than asserted:

  - **no float tolerance at all** — `set()` uses `==`/hash on raw values,
    so `2.9999999999` and `3.0` are different set elements.
  - **duplicates collapse**: `[(1,), (1,), (2,)]` and `[(1,), (2,)]` are
    the same set. Our comparator's default `duplicate_row_mode="multiset"`
    treats these as NOT_EQUAL; `duplicate_row_mode="set"` reproduces this
    exact BIRD behavior for the consistency-rate comparison.
  - **row order never matters** — `set()` has none, regardless of whether
    the gold SQL has an ORDER BY. Our comparator's `row_order="ordered"`
    path is strictly more correct when the gold query does specify order.
  - **column order matters** unless the SELECT lists happen to align —
    `set()` of row tuples is positional, same as our `column_match_mode=
    "by_position"` default.
  - **every execution error becomes "wrong"**, indistinguishable from a
    semantically incorrect but successfully-executing query — this is
    exactly the model-error-vs-harness-error conflation CLAUDE.md §2.3
    calls the "most easily made, hardest to find, most consequential" bug
    class, ported faithfully here so the consistency-rate script can show
    it concretely rather than assert it.
"""

from __future__ import annotations

import sqlite3


def bird_official_execute_sql(
    predicted_sql: str, ground_truth_sql: str, db_path: str, *, connect_timeout_s: float = 30.0
) -> int:
    """Verbatim port of BIRD's `execute_sql` (see module docstring for
    source). Returns 1 if `set(predicted) == set(ground_truth)`, else 0.

    One deliberate deviation from the original: `connect_timeout_s` is
    passed to `sqlite3.connect()` (the real script omits it entirely).
    This only bounds lock-contention waits, not a CPU-bound runaway query
    (same limitation CLAUDE.md/A2 identified for `sqlite3.connect`'s own
    timeout kwarg — the real script's `func_timeout` wrapper has the same
    class of weakness for C-extension calls). Since this function is used
    against curated golden-set/gold SQL, not adversarial LLM output, and
    the real A2 sandbox (process-isolated, actually kills on timeout) is
    what production eval/reward paths use instead of this file, closing
    the "zero timeout at all" gap is worth it without chasing full
    process isolation here too.

    Otherwise reproduces the original's behavior of letting *any*
    exception from the predicted SQL propagate uncaught — the real script
    relies on its caller (`execute_model`, using `func_timeout`) to catch
    it and score 0. We mirror that split here: this function raises; use
    `bird_official_compare` for the "always returns 0/1, errors become 0"
    behavior actually used for scoring.
    """
    conn = sqlite3.connect(db_path, timeout=connect_timeout_s)
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        return 1 if set(predicted_res) == set(ground_truth_res) else 0
    finally:
        conn.close()


def bird_official_compare(predicted_sql: str, ground_truth_sql: str, db_path: str) -> int:
    """The actual scoring behavior: any exception (bad SQL, timeout,
    missing table, ...) becomes 0, same as `execute_model`'s except
    clauses in the real script. This is what a "consistency rate" against
    our own comparator should be measured against."""
    try:
        return bird_official_execute_sql(predicted_sql, ground_truth_sql, db_path)
    except Exception:
        # check-no-cheating: allow=EXCEPT_RETURN_CONSTANT reason=faithful port, see docstring
        return 0
