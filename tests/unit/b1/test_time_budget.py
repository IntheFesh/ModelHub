"""Unit tests for train/time_budget.py — uses an injectable `now_fn`
(matching gateway/circuit_breaker.py's precedent) so budget-exhaustion
transitions are tested deterministically, no wall-clock sleeping."""

from __future__ import annotations

import pytest

from modelhub.train.time_budget import TimeBudgetTracker


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_hours(self, hours: float) -> None:
        self.now += hours * 3600


class TestTimeBudgetTracker:
    def test_not_exhausted_before_max_hours(self) -> None:
        clock = _FakeClock()
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=clock)
        clock.advance_hours(7.9)
        assert tracker.budget_exhausted is False

    def test_exhausted_at_exactly_max_hours(self) -> None:
        clock = _FakeClock()
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=clock)
        clock.advance_hours(8.0)
        assert tracker.budget_exhausted is True

    def test_exhausted_past_max_hours(self) -> None:
        clock = _FakeClock()
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=clock)
        clock.advance_hours(8.1)
        assert tracker.budget_exhausted is True

    def test_elapsed_hours_tracks_the_fake_clock(self) -> None:
        clock = _FakeClock()
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=clock)
        clock.advance_hours(2.5)
        assert tracker.elapsed_hours == pytest.approx(2.5)

    def test_non_positive_max_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            TimeBudgetTracker(max_hours=0.0, now_fn=_FakeClock())


class TestRecordMetric:
    def test_first_metric_is_always_best(self) -> None:
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=_FakeClock())
        assert tracker.record_metric(1, 0.5) is True
        assert tracker.best_step == 1
        assert tracker.best_metric == 0.5

    def test_higher_is_better_by_default(self) -> None:
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=_FakeClock())
        tracker.record_metric(1, 0.5)
        assert tracker.record_metric(2, 0.4) is False
        assert tracker.record_metric(3, 0.9) is True
        assert tracker.best_step == 3
        assert tracker.best_metric == 0.9

    def test_lower_is_better_when_requested(self) -> None:
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=_FakeClock())
        tracker.record_metric(1, 1.0, higher_is_better=False)
        assert tracker.record_metric(2, 0.3, higher_is_better=False) is True
        assert tracker.record_metric(3, 0.8, higher_is_better=False) is False
        assert tracker.best_step == 2
        assert tracker.best_metric == 0.3

    def test_no_metrics_recorded_yet(self) -> None:
        tracker = TimeBudgetTracker(max_hours=8.0, now_fn=_FakeClock())
        assert tracker.best_step is None
        assert tracker.best_metric is None
