"""Smoke test: a full overnight cycle at shrunk scale — a real
overprovisioned queue (PLAN.md ★ item 5) where the time budget runs out
partway through, real risk-ascending ordering, a real smoke-test-skip,
a real heartbeat write, and real per-task manifests, all in one run
(CLAUDE.md §1.4 — every real code path runs; only the number of tasks
and the time budget are shrunk to smoke scale)."""

from __future__ import annotations

from pathlib import Path

import night_queue
from modelhub.common.capability import CheckStatus


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_overprovisioned_queue_with_rollover_and_smoke_skip(tmp_path: Path) -> None:
    clock = _FakeClock()
    completed_calls: list[str] = []

    def _make_run(task_id: str, *, duration_s: float) -> object:
        def _run() -> None:
            completed_calls.append(task_id)
            clock.now += duration_s

        return _run

    tasks = [
        night_queue.NightTask(
            task_id="low-risk-check",
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=10.0,
            run=_make_run("low-risk-check", duration_s=10.0),
        ),
        night_queue.NightTask(
            task_id="skipped-by-smoke-test",
            risk_level=night_queue.TaskRiskLevel.MEDIUM,
            estimated_duration_s=10.0,
            run=_make_run("skipped-by-smoke-test", duration_s=10.0),
            smoke_test=lambda: CheckStatus.FAIL,
        ),
        night_queue.NightTask(
            task_id="high-risk-training",
            risk_level=night_queue.TaskRiskLevel.HIGH,
            estimated_duration_s=10.0,
            run=_make_run("high-risk-training", duration_s=10.0),
        ),
        # overprovisioned past what the shrunk time budget allows —
        # this one must come back DEFERRED, not silently dropped.
        night_queue.NightTask(
            task_id="rolled-over-tomorrow",
            risk_level=night_queue.TaskRiskLevel.HIGH,
            estimated_duration_s=10.0,
            run=_make_run("rolled-over-tomorrow", duration_s=10.0),
        ),
    ]

    queue_date = "20260824-smoke"
    snapshot_path = tmp_path / f"{queue_date}.json"
    night_queue.write_queue_snapshot(tasks, queue_date=queue_date, path=snapshot_path)
    assert snapshot_path.is_file()

    manifests_dir = tmp_path / "manifests"
    heartbeat_path = tmp_path / "heartbeat.json"
    report = night_queue.run_night_queue(
        tasks,
        # budget for exactly 2 tasks' worth of real work (skipped task
        # costs ~0s since it never runs) — by the time the 3rd real
        # task would start, elapsed time already meets the budget,
        # forcing a real DEFERRED rollover for the 4th.
        time_budget_s=15.0,
        heartbeat_path=heartbeat_path,
        manifests_dir=manifests_dir,
        now_fn=clock,
        now_iso_fn=lambda: "2026-08-24T00:00:00Z",
        queue_date=queue_date,
    )

    outcomes_by_id = {o.task_id: o for o in report.outcomes}
    # risk-ascending order was honored: low-risk and the smoke-skipped
    # medium-risk task both got a chance before either high-risk task.
    assert outcomes_by_id["low-risk-check"].status is night_queue.QueueTaskStatus.COMPLETED
    assert (
        outcomes_by_id["skipped-by-smoke-test"].status
        is night_queue.QueueTaskStatus.SMOKE_TEST_NOT_PASSED
    )
    assert "skipped-by-smoke-test" not in completed_calls
    assert outcomes_by_id["high-risk-training"].status is night_queue.QueueTaskStatus.COMPLETED
    assert outcomes_by_id["rolled-over-tomorrow"].status is night_queue.QueueTaskStatus.DEFERRED

    # a completed task has a real manifest; the deferred/skipped ones do not.
    assert (manifests_dir / "low-risk-check.json").is_file()
    assert (manifests_dir / "high-risk-training.json").is_file()
    assert not (manifests_dir / "skipped-by-smoke-test.json").exists()
    assert not (manifests_dir / "rolled-over-tomorrow.json").exists()

    assert heartbeat_path.is_file()
