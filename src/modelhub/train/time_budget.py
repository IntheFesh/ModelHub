"""Time-budget-based training control — PLAN.md B1: "按时间预算跑不按
epoch 跑：--max-hours，到点取最佳 checkpoint."

The clock is injectable (`now_fn`), matching `gateway/circuit_breaker.py`'s
precedent — tests exercise real budget-exhaustion transitions without
sleeping in wall-clock time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TimeBudgetTracker:
    max_hours: float
    now_fn: Callable[[], float] = time.monotonic
    _start_time: float = field(init=False)
    _best_metric: float | None = field(default=None, init=False)
    _best_step: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.max_hours <= 0:
            raise ValueError(f"max_hours must be positive, got {self.max_hours}")
        self._start_time = self.now_fn()

    @property
    def elapsed_hours(self) -> float:
        return (self.now_fn() - self._start_time) / 3600

    @property
    def budget_exhausted(self) -> bool:
        return self.elapsed_hours >= self.max_hours

    def record_metric(self, step: int, metric: float, *, higher_is_better: bool = True) -> bool:
        """Record `metric` at `step`; returns True iff this is the new
        best-so-far (by `higher_is_better`'s direction)."""
        is_best = (
            self._best_metric is None
            or (higher_is_better and metric > self._best_metric)
            or (not higher_is_better and metric < self._best_metric)
        )
        if is_best:
            self._best_metric = metric
            self._best_step = step
        return is_best

    @property
    def best_step(self) -> int | None:
        return self._best_step

    @property
    def best_metric(self) -> float | None:
        return self._best_metric


__all__ = ["TimeBudgetTracker"]
