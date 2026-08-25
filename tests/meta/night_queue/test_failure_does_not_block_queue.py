"""CLAUDE.md §1.5 required meta-test — and this round's own explicit
acceptance criterion (PLAN.md: "验收：模拟一个必然失败的任务在队列
中间，断言后续任务照常执行且状态正确记录"): a deliberately-always-
failing task placed in the middle of the queue must not block the tasks
after it, and every task's status must be correctly recorded — real `;`
semantics, not `&&`. Also proves the mechanism is not permanently
"queue halts on first failure" by running an all-healthy queue end to
end."""

from __future__ import annotations

from pathlib import Path

import night_queue
from modelhub.common.capability import CheckStatus


def _noop() -> None:
    pass


def _always_fails() -> None:
    raise RuntimeError("this task must always fail")


def _smoke_test_always_raises() -> CheckStatus:
    raise RuntimeError("smoke probe itself is broken")


def test_a_guaranteed_failure_in_the_middle_does_not_block_later_tasks(tmp_path: Path) -> None:
    calls: list[str] = []

    def _tracked(task_id: str) -> night_queue.NightTask:
        def _run() -> None:
            calls.append(task_id)

        return night_queue.NightTask(
            task_id=task_id,
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=1.0,
            run=_run,
        )

    tasks = [
        _tracked("before-1"),
        _tracked("before-2"),
        night_queue.NightTask(
            task_id="the-guaranteed-failure",
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=1.0,
            run=_always_fails,
        ),
        _tracked("after-1"),
        _tracked("after-2"),
    ]

    report = night_queue.run_night_queue(
        tasks,
        time_budget_s=3600.0,
        heartbeat_path=tmp_path / "heartbeat.json",
        now_iso_fn=lambda: "2026-08-24T00:00:00Z",
    )

    # every task after the failure genuinely ran (not skipped, not
    # silently dropped by an exception propagating out of the loop).
    assert "after-1" in calls
    assert "after-2" in calls
    assert len(calls) == 4  # every task except the failing one itself

    # status is correctly recorded for every single task, not just
    # "the queue didn't crash".
    outcomes_by_id = {o.task_id: o for o in report.outcomes}
    assert outcomes_by_id["before-1"].status is night_queue.QueueTaskStatus.COMPLETED
    assert outcomes_by_id["before-2"].status is night_queue.QueueTaskStatus.COMPLETED
    assert outcomes_by_id["the-guaranteed-failure"].status is night_queue.QueueTaskStatus.FAILED
    assert "this task must always fail" in (outcomes_by_id["the-guaranteed-failure"].detail or "")
    assert outcomes_by_id["after-1"].status is night_queue.QueueTaskStatus.COMPLETED
    assert outcomes_by_id["after-2"].status is night_queue.QueueTaskStatus.COMPLETED

    assert len(report.outcomes) == 5  # nothing silently dropped from the report either


def test_a_smoke_test_that_raises_in_the_middle_does_not_block_later_tasks(
    tmp_path: Path,
) -> None:
    # This round's own acceptance criterion, applied to the failure mode a
    # raising smoke_test represents (distinct from task.run() raising,
    # which the test above already covers) — a real bug this pass found:
    # the probe call used to sit outside run_task_with_watchdog's
    # try/except entirely, so this would have propagated straight out of
    # run_night_queue and killed the whole overnight run.
    calls: list[str] = []

    def _tracked(task_id: str) -> night_queue.NightTask:
        def _run() -> None:
            calls.append(task_id)

        return night_queue.NightTask(
            task_id=task_id,
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=1.0,
            run=_run,
        )

    tasks = [
        _tracked("before-1"),
        night_queue.NightTask(
            task_id="the-broken-smoke-probe",
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=1.0,
            run=_noop,
            smoke_test=_smoke_test_always_raises,
        ),
        _tracked("after-1"),
    ]

    report = night_queue.run_night_queue(
        tasks,
        time_budget_s=3600.0,
        heartbeat_path=tmp_path / "heartbeat.json",
        now_iso_fn=lambda: "2026-08-24T00:00:00Z",
    )

    assert "before-1" in calls
    assert "after-1" in calls

    outcomes_by_id = {o.task_id: o for o in report.outcomes}
    assert outcomes_by_id["before-1"].status is night_queue.QueueTaskStatus.COMPLETED
    assert outcomes_by_id["the-broken-smoke-probe"].status is night_queue.QueueTaskStatus.FAILED
    assert outcomes_by_id["after-1"].status is night_queue.QueueTaskStatus.COMPLETED
    assert len(report.outcomes) == 3


def test_an_all_healthy_queue_is_not_permanently_blocked_by_the_mechanism(
    tmp_path: Path,
) -> None:
    tasks = [
        night_queue.NightTask(
            task_id=f"t{i}",
            risk_level=night_queue.TaskRiskLevel.LOW,
            estimated_duration_s=1.0,
            run=_noop,
        )
        for i in range(4)
    ]
    report = night_queue.run_night_queue(
        tasks,
        time_budget_s=3600.0,
        heartbeat_path=tmp_path / "heartbeat.json",
        now_iso_fn=lambda: "2026-08-24T00:00:00Z",
    )
    assert set(report.completed) == {"t0", "t1", "t2", "t3"}
    assert report.failed == ()
