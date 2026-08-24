#!/usr/bin/env python3
"""`make night-queue` — unattended overnight task orchestration
(CLAUDE.md §6.2).

Requirements, each mapped directly onto a piece of this module:

1. Risk-ascending ordering, deterministic tasks (quantization, eval)
   first — `build_night_queue`'s stable sort by `TaskRiskLevel`.
2. `;` semantics, not `&&` — one task's failure never blocks the rest
   of the queue: `run_task_with_watchdog` catches every real exception
   and returns a `FAILED` outcome rather than propagating it, so
   `run_night_queue`'s loop always reaches the next task.
3. Every task that actually ran gets its own manifest with
   `status in {COMPLETED, FAILED, INTERRUPTED}` — `write_task_manifest`.
   A task that never ran (smoke-test-skipped or deferred) gets no
   per-task manifest; its disposition lives only in the queue snapshot
   (item 6) — there is nothing a manifest could honestly report about a
   run that never happened.
4. ★ Watchdog: no artificial delay between tasks (the loop moves on the
   instant one task's `run()` returns, raises, or its own
   `watchdog_timeout_s` safety-net timeout fires) — GPU never sits idle
   waiting on this orchestrator. The per-task timeout is a real
   `ThreadPoolExecutor` + `Future.result(timeout=...)`, the same
   concurrency primitive `sqlexec/pool.py` already uses for the
   identical "don't let one task hang everything" problem.
5. ★ Overprovisioning + rollover: `run_night_queue` takes a
   `time_budget_s`; once elapsed time reaches it, every remaining
   queued task is marked `DEFERRED` (not run, not silently dropped) —
   see `main()`'s queue-building comment for how a caller overprovisions
   30-50% past the estimated nightly capacity.
6. Full queue + duration estimate printed and persisted before any task
   runs — `render_queue_snapshot`/`write_queue_snapshot`, written to
   `artifacts/queues/<date>.json`.
7. ★ Optional per-task smoke-test hook returns a real `CheckStatus`
   (PASS/FAIL/SKIP) — only `PASS` lets the real task run.
   `SMOKE_TEST_NOT_PASSED` covers both FAIL and SKIP, and is never the
   same status as `COMPLETED` (CLAUDE.md §1.3/§1.5's
   `test_skip_is_not_pass`, applied here directly rather than only in a
   dependency-detection context).
8. Heartbeat every ~15 minutes — `render_heartbeat`/`write_heartbeat`,
   invoked from `run_night_queue`'s loop by real elapsed wall-clock
   time, plus one final heartbeat write after the last task so the
   queue's end state is always on disk.

Acceptance (this round's own explicit ask): a deliberately-always-
failing task placed in the middle of a queue must not block the tasks
after it, and every task's status must be correctly recorded — see
`tests/meta/night_queue/test_failure_does_not_block_queue.py`.
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from modelhub.common.atomic_io import atomic_write_json
from modelhub.common.capability import CheckStatus

_HEARTBEAT_INTERVAL_S = 15 * 60.0


class TaskRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_RISK_ORDER: dict[TaskRiskLevel, int] = {
    TaskRiskLevel.LOW: 0,
    TaskRiskLevel.MEDIUM: 1,
    TaskRiskLevel.HIGH: 2,
}


class QueueTaskStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    SMOKE_TEST_NOT_PASSED = "SMOKE_TEST_NOT_PASSED"
    DEFERRED = "DEFERRED"


@dataclass(frozen=True)
class NightTask:
    task_id: str
    risk_level: TaskRiskLevel
    estimated_duration_s: float
    run: Callable[[], None]
    smoke_test: Callable[[], CheckStatus] | None = None
    watchdog_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.estimated_duration_s <= 0:
            raise ValueError(
                f"{self.task_id}: estimated_duration_s must be positive, "
                f"got {self.estimated_duration_s}"
            )
        if self.watchdog_timeout_s is not None and self.watchdog_timeout_s <= 0:
            raise ValueError(
                f"{self.task_id}: watchdog_timeout_s must be positive, "
                f"got {self.watchdog_timeout_s}"
            )


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    status: QueueTaskStatus
    started_at: float | None
    ended_at: float | None
    detail: str | None


@dataclass(frozen=True)
class NightQueueReport:
    outcomes: tuple[TaskOutcome, ...]

    def _by_status(self, status: QueueTaskStatus) -> tuple[str, ...]:
        return tuple(o.task_id for o in self.outcomes if o.status is status)

    @property
    def completed(self) -> tuple[str, ...]:
        return self._by_status(QueueTaskStatus.COMPLETED)

    @property
    def failed(self) -> tuple[str, ...]:
        return self._by_status(QueueTaskStatus.FAILED)

    @property
    def deferred(self) -> tuple[str, ...]:
        return self._by_status(QueueTaskStatus.DEFERRED)


def build_night_queue(tasks: Sequence[NightTask]) -> list[NightTask]:
    """Item 1: risk-ascending, stable (equal-risk tasks keep the caller's
    relative order — "确定性任务先跑" only says LOW-risk goes first, not
    that same-risk tasks need reordering among themselves)."""
    return sorted(tasks, key=lambda t: _RISK_ORDER[t.risk_level])


def run_task_with_watchdog(
    task: NightTask, *, now_fn: Callable[[], float] = time.monotonic
) -> TaskOutcome:
    if task.smoke_test is not None:
        smoke_status = task.smoke_test()
        if smoke_status is not CheckStatus.PASS:
            # ★ item 7: SKIP is not the same outcome as PASS — a FAIL
            # and a SKIP smoke test both mean "did not run the real
            # task", both get the identical honest status.
            return TaskOutcome(
                task_id=task.task_id,
                status=QueueTaskStatus.SMOKE_TEST_NOT_PASSED,
                started_at=None,
                ended_at=None,
                detail=f"smoke_test returned {smoke_status.value}, not PASS",
            )

    started_at = now_fn()
    try:
        if task.watchdog_timeout_s is None:
            task.run()
        else:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(task.run).result(timeout=task.watchdog_timeout_s)
    except KeyboardInterrupt:
        return TaskOutcome(
            task_id=task.task_id,
            status=QueueTaskStatus.INTERRUPTED,
            started_at=started_at,
            ended_at=now_fn(),
            detail="interrupted (SIGINT/SIGTERM) while running",
        )
    except FutureTimeoutError:
        return TaskOutcome(
            task_id=task.task_id,
            status=QueueTaskStatus.FAILED,
            started_at=started_at,
            ended_at=now_fn(),
            detail=f"exceeded watchdog_timeout_s={task.watchdog_timeout_s}",
        )
    except Exception as e:
        # failure (any exception) must never stop the queue; the real
        # exception text is preserved in `detail`, not swallowed.
        return TaskOutcome(
            task_id=task.task_id,
            status=QueueTaskStatus.FAILED,
            started_at=started_at,
            ended_at=now_fn(),
            detail=f"{type(e).__name__}: {e}",
        )
    return TaskOutcome(
        task_id=task.task_id,
        status=QueueTaskStatus.COMPLETED,
        started_at=started_at,
        ended_at=now_fn(),
        detail=None,
    )


def render_queue_snapshot(queue: Sequence[NightTask], *, queue_date: str) -> dict[str, Any]:
    total_estimated_s = sum(t.estimated_duration_s for t in queue)
    return {
        "queue_date": queue_date,
        "task_count": len(queue),
        "total_estimated_duration_s": total_estimated_s,
        "tasks": [
            {
                "task_id": t.task_id,
                "risk_level": t.risk_level.value,
                "estimated_duration_s": t.estimated_duration_s,
                "has_smoke_test": t.smoke_test is not None,
            }
            for t in queue
        ],
    }


def write_queue_snapshot(queue: Sequence[NightTask], *, queue_date: str, path: Path) -> Path:
    return atomic_write_json(path, render_queue_snapshot(queue, queue_date=queue_date))


_MANIFEST_STATUSES = frozenset(
    {QueueTaskStatus.COMPLETED, QueueTaskStatus.FAILED, QueueTaskStatus.INTERRUPTED}
)


def write_task_manifest(outcome: TaskOutcome, *, path: Path) -> Path | None:
    """Item 3: "每任务独立写 manifest，status ∈ {COMPLETED, FAILED,
    INTERRUPTED}" — only for a task that actually started; a task that
    never ran (`SMOKE_TEST_NOT_PASSED`/`DEFERRED`) has nothing a
    manifest could honestly report, so this returns `None` rather than
    writing a manifest with a status outside PLAN.md's literal set."""
    if outcome.status not in _MANIFEST_STATUSES:
        return None
    return atomic_write_json(
        path,
        {
            "task_id": outcome.task_id,
            "status": outcome.status.value,
            "started_at": outcome.started_at,
            "ended_at": outcome.ended_at,
            "detail": outcome.detail,
        },
    )


def render_heartbeat(
    report: NightQueueReport, *, queue_date: str, written_at: str
) -> dict[str, Any]:
    return {
        "queue_date": queue_date,
        "written_at": written_at,
        "task_count": len(report.outcomes),
        "outcomes": [
            {
                "task_id": o.task_id,
                "status": o.status.value,
                "started_at": o.started_at,
                "ended_at": o.ended_at,
                "detail": o.detail,
            }
            for o in report.outcomes
        ],
    }


def write_heartbeat(
    report: NightQueueReport, *, queue_date: str, path: Path, now_iso_fn: Callable[[], str]
) -> Path:
    heartbeat = render_heartbeat(report, queue_date=queue_date, written_at=now_iso_fn())
    return atomic_write_json(path, heartbeat)


def run_night_queue(
    tasks: Sequence[NightTask],
    *,
    time_budget_s: float,
    heartbeat_path: Path,
    manifests_dir: Path | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    now_iso_fn: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
    heartbeat_interval_s: float = _HEARTBEAT_INTERVAL_S,
    queue_date: str = "unknown",
) -> NightQueueReport:
    if time_budget_s <= 0:
        raise ValueError(f"time_budget_s must be positive, got {time_budget_s}")
    queue = build_night_queue(tasks)
    outcomes: list[TaskOutcome] = []
    start = now_fn()
    last_heartbeat = start
    interrupted = False

    for task in queue:
        if interrupted or (now_fn() - start) >= time_budget_s:
            outcomes.append(
                TaskOutcome(
                    task_id=task.task_id,
                    status=QueueTaskStatus.DEFERRED,
                    started_at=None,
                    ended_at=None,
                    detail="time budget exhausted (or queue interrupted) before this task started"
                    if not interrupted
                    else "queue interrupted before this task started",
                )
            )
            continue

        outcome = run_task_with_watchdog(task, now_fn=now_fn)
        outcomes.append(outcome)
        if manifests_dir is not None:
            write_task_manifest(outcome, path=manifests_dir / f"{task.task_id}.json")
        if outcome.status is QueueTaskStatus.INTERRUPTED:
            # item: an interrupted queue does not proceed to the next
            # task — the whole process is shutting down, not just one
            # task failing.
            interrupted = True

        if (now_fn() - last_heartbeat) >= heartbeat_interval_s:
            write_heartbeat(
                NightQueueReport(outcomes=tuple(outcomes)),
                queue_date=queue_date,
                path=heartbeat_path,
                now_iso_fn=now_iso_fn,
            )
            last_heartbeat = now_fn()

    report = NightQueueReport(outcomes=tuple(outcomes))
    write_heartbeat(report, queue_date=queue_date, path=heartbeat_path, now_iso_fn=now_iso_fn)
    return report


def _install_sigterm_as_keyboard_interrupt() -> None:
    """SIGTERM (the real signal an unattended overnight job receives on
    shutdown/preemption) is converted to the same `KeyboardInterrupt`
    Python already raises for SIGINT, so `run_task_with_watchdog`'s
    single interrupt-handling branch covers both real signals."""

    def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def main() -> int:
    _install_sigterm_as_keyboard_interrupt()

    from night_queue_fixtures.example_tasks import build_example_night_tasks

    tasks = build_example_night_tasks()
    queue_date = datetime.now(UTC).strftime("%Y%m%d")
    queue = build_night_queue(tasks)

    print("=" * 72)
    print(f"night_queue — {len(queue)} task(s) queued for {queue_date}")
    print("=" * 72)
    for t in queue:
        print(f"  [{t.risk_level.value:>6}] {t.task_id} (~{t.estimated_duration_s:.0f}s)")
    snapshot_path = Path("artifacts/queues") / f"{queue_date}.json"
    write_queue_snapshot(queue, queue_date=queue_date, path=snapshot_path)
    print(f"queue snapshot -> {snapshot_path}")

    heartbeat_path = Path("artifacts/queues") / f"{queue_date}-heartbeat.json"
    manifests_dir = Path("artifacts/queues") / queue_date / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    report = run_night_queue(
        tasks,
        # ★ item 5: overprovision the real queue 30-50% past one
        # night's estimated capacity when building `tasks` (this
        # example queue is small enough not to need it) — whatever
        # doesn't fit in time_budget_s comes back as DEFERRED, not lost.
        time_budget_s=8 * 3600.0,
        heartbeat_path=heartbeat_path,
        manifests_dir=manifests_dir,
        queue_date=queue_date,
    )

    print("\n" + "=" * 72)
    print(
        f"completed={len(report.completed)} failed={len(report.failed)} "
        f"deferred={len(report.deferred)}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
