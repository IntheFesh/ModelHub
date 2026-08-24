"""Bounded-concurrency batch execution on top of `sandbox.execute_isolated`.

Each task still gets its own OS process (that is what makes a timeout an
actual kill rather than best-effort cooperation); this module only bounds
how many of those processes run at once, so a batch of thousands of
queries does not try to fork thousands of processes simultaneously.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from modelhub.sqlexec.backends import DbRef
from modelhub.sqlexec.outcomes import ExecOutcome
from modelhub.sqlexec.sandbox import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_ROW_LIMIT,
    DEFAULT_TIMEOUT_S,
    execute_isolated,
)


@dataclass(frozen=True)
class SqlTask:
    task_id: str
    ref: DbRef
    sql: str


def execute_many(
    tasks: Sequence[SqlTask],
    *,
    max_concurrency: int = 8,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> dict[str, ExecOutcome]:
    """Run every task, at most `max_concurrency` at a time. Returns task_id -> outcome.

    One task's outcome never depends on another's: a slow/killed/crashed
    task cannot strand or block the rest of the batch, since each runs in
    its own process managed by its own thread here.
    """
    if not tasks:
        return {}

    results: dict[str, ExecOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_id = {
            pool.submit(
                execute_isolated,
                task.ref,
                task.sql,
                row_limit=row_limit,
                timeout_s=timeout_s,
                memory_limit_mb=memory_limit_mb,
            ): task.task_id
            for task in tasks
        }
        for future in future_to_id:
            task_id = future_to_id[future]
            results[task_id] = future.result()
    return results
