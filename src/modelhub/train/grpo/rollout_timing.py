"""PLAN.md item 6: "rollout 的 SQL 执行要并发且有超时，否则 rollout
成为训练瓶颈。★ 测 rollout 时间里有多少花在等 SQL 上——这个数字很值钱".

Concurrency+timeout is not reimplemented here — `sqlexec.pool.
execute_many` (A2) already bounds concurrency and enforces a real
per-task timeout via `execute_isolated`'s subprocess kill. This module's
only job is wrapping that call with wall-clock timing so the "SQL wait
share" PLAN.md calls out as valuable is a real measured number, not an
estimate.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from modelhub.sqlexec.outcomes import ExecOutcome
from modelhub.sqlexec.pool import SqlTask, execute_many


@dataclass(frozen=True)
class RolloutTimingBreakdown:
    total_rollout_s: float
    sql_wait_s: float

    def __post_init__(self) -> None:
        if self.total_rollout_s <= 0:
            raise ValueError(f"total_rollout_s must be positive, got {self.total_rollout_s}")
        if self.sql_wait_s < 0:
            raise ValueError(f"sql_wait_s must be non-negative, got {self.sql_wait_s}")
        if self.sql_wait_s > self.total_rollout_s:
            raise ValueError(
                f"sql_wait_s ({self.sql_wait_s}) cannot exceed total_rollout_s "
                f"({self.total_rollout_s}) — SQL execution is a subset of the rollout, "
                f"not the whole thing"
            )

    @property
    def sql_wait_share(self) -> float:
        return self.sql_wait_s / self.total_rollout_s


def execute_rollout_sql_batch(
    tasks: Sequence[SqlTask],
    *,
    max_concurrency: int,
    timeout_s: float,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, ExecOutcome], float]:
    """Real concurrent+timeout SQL execution (`sqlexec.pool.execute_many`
    unchanged) with real wall-clock timing around the call. Returns
    `(outcomes, sql_wait_s)` — the caller (which also knows the total
    rollout wall time: generation + this) builds the
    `RolloutTimingBreakdown`."""
    if not tasks:
        raise ValueError("cannot execute a rollout SQL batch with zero tasks")
    start = now_fn()
    outcomes = execute_many(tasks, max_concurrency=max_concurrency, timeout_s=timeout_s)
    elapsed = now_fn() - start
    return outcomes, elapsed


__all__ = ["RolloutTimingBreakdown", "execute_rollout_sql_batch"]
