"""Unit tests for scripts/night_queue.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import night_queue
from modelhub.common.capability import CheckStatus


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _noop() -> None:
    pass


def _always_raises() -> None:
    raise ValueError("simulated task failure")


def _task(task_id: str, **overrides: object) -> night_queue.NightTask:
    defaults: dict[str, object] = {
        "task_id": task_id,
        "risk_level": night_queue.TaskRiskLevel.LOW,
        "estimated_duration_s": 10.0,
        "run": _noop,
    }
    defaults.update(overrides)
    return night_queue.NightTask(**defaults)  # type: ignore[arg-type]


class TestNightTaskValidation:
    def test_non_positive_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="estimated_duration_s"):
            _task("t0", estimated_duration_s=0.0)

    def test_non_positive_watchdog_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="watchdog_timeout_s"):
            _task("t0", watchdog_timeout_s=0.0)


class TestBuildNightQueue:
    def test_sorts_by_risk_ascending(self) -> None:
        high = _task("high", risk_level=night_queue.TaskRiskLevel.HIGH)
        low = _task("low", risk_level=night_queue.TaskRiskLevel.LOW)
        medium = _task("medium", risk_level=night_queue.TaskRiskLevel.MEDIUM)
        queue = night_queue.build_night_queue([high, medium, low])
        assert [t.task_id for t in queue] == ["low", "medium", "high"]

    def test_stable_within_same_risk_level(self) -> None:
        a = _task("a", risk_level=night_queue.TaskRiskLevel.LOW)
        b = _task("b", risk_level=night_queue.TaskRiskLevel.LOW)
        queue = night_queue.build_night_queue([a, b])
        assert [t.task_id for t in queue] == ["a", "b"]


class TestRunTaskWithWatchdog:
    def test_healthy_task_completes(self) -> None:
        clock = _FakeClock()
        outcome = night_queue.run_task_with_watchdog(_task("t0"), now_fn=clock)
        assert outcome.status is night_queue.QueueTaskStatus.COMPLETED
        assert outcome.started_at is not None
        assert outcome.detail is None

    def test_raising_task_is_failed_not_propagated(self) -> None:
        clock = _FakeClock()
        outcome = night_queue.run_task_with_watchdog(_task("t0", run=_always_raises), now_fn=clock)
        assert outcome.status is night_queue.QueueTaskStatus.FAILED
        assert "simulated task failure" in (outcome.detail or "")

    def test_keyboard_interrupt_is_interrupted(self) -> None:
        def _interrupt() -> None:
            raise KeyboardInterrupt

        clock = _FakeClock()
        outcome = night_queue.run_task_with_watchdog(_task("t0", run=_interrupt), now_fn=clock)
        assert outcome.status is night_queue.QueueTaskStatus.INTERRUPTED

    def test_smoke_test_pass_runs_the_real_task(self) -> None:
        calls = {"ran": False}

        def _run() -> None:
            calls["ran"] = True

        outcome = night_queue.run_task_with_watchdog(
            _task("t0", run=_run, smoke_test=lambda: CheckStatus.PASS)
        )
        assert calls["ran"] is True
        assert outcome.status is night_queue.QueueTaskStatus.COMPLETED

    def test_smoke_test_fail_skips_the_real_task(self) -> None:
        calls = {"ran": False}

        def _run() -> None:
            calls["ran"] = True

        outcome = night_queue.run_task_with_watchdog(
            _task("t0", run=_run, smoke_test=lambda: CheckStatus.FAIL)
        )
        assert calls["ran"] is False
        assert outcome.status is night_queue.QueueTaskStatus.SMOKE_TEST_NOT_PASSED

    def test_smoke_test_skip_is_not_pass(self) -> None:
        """CLAUDE.md §1.3/§1.5: SKIP must never read as COMPLETED."""
        calls = {"ran": False}

        def _run() -> None:
            calls["ran"] = True

        outcome = night_queue.run_task_with_watchdog(
            _task("t0", run=_run, smoke_test=lambda: CheckStatus.SKIP)
        )
        assert calls["ran"] is False
        assert outcome.status is night_queue.QueueTaskStatus.SMOKE_TEST_NOT_PASSED
        assert outcome.status is not night_queue.QueueTaskStatus.COMPLETED

    def test_watchdog_timeout_marks_failed(self) -> None:
        import time

        def _slow() -> None:
            time.sleep(0.3)

        outcome = night_queue.run_task_with_watchdog(
            _task("t0", run=_slow, watchdog_timeout_s=0.05)
        )
        assert outcome.status is night_queue.QueueTaskStatus.FAILED
        assert "watchdog_timeout_s" in (outcome.detail or "")


class TestQueueSnapshot:
    def test_write_and_read_back(self, tmp_path: Path) -> None:
        queue = [_task("t0"), _task("t1", risk_level=night_queue.TaskRiskLevel.HIGH)]
        path = tmp_path / "snapshot.json"
        night_queue.write_queue_snapshot(queue, queue_date="20260824", path=path)
        payload = json.loads(path.read_text())
        assert payload["task_count"] == 2
        assert payload["queue_date"] == "20260824"
        assert payload["total_estimated_duration_s"] == 20.0


class TestWriteTaskManifest:
    def _outcome(self, **overrides: object) -> night_queue.TaskOutcome:
        defaults: dict[str, object] = {
            "task_id": "t0",
            "status": night_queue.QueueTaskStatus.COMPLETED,
            "started_at": 1.0,
            "ended_at": 2.0,
            "detail": None,
        }
        defaults.update(overrides)
        return night_queue.TaskOutcome(**defaults)  # type: ignore[arg-type]

    def test_completed_writes_a_real_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "t0.json"
        result = night_queue.write_task_manifest(self._outcome(), path=path)
        assert result == path
        payload = json.loads(path.read_text())
        assert payload["status"] == "COMPLETED"

    def test_failed_writes_a_real_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "t0.json"
        outcome = self._outcome(status=night_queue.QueueTaskStatus.FAILED, detail="boom")
        night_queue.write_task_manifest(outcome, path=path)
        payload = json.loads(path.read_text())
        assert payload["status"] == "FAILED"
        assert payload["detail"] == "boom"

    def test_smoke_test_not_passed_writes_no_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "t0.json"
        outcome = self._outcome(
            status=night_queue.QueueTaskStatus.SMOKE_TEST_NOT_PASSED,
            started_at=None,
            ended_at=None,
        )
        result = night_queue.write_task_manifest(outcome, path=path)
        assert result is None
        assert not path.exists()

    def test_deferred_writes_no_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "t0.json"
        outcome = self._outcome(
            status=night_queue.QueueTaskStatus.DEFERRED, started_at=None, ended_at=None
        )
        result = night_queue.write_task_manifest(outcome, path=path)
        assert result is None
        assert not path.exists()


class TestRunNightQueue:
    def test_all_tasks_complete_within_budget(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        tasks = [_task("t0"), _task("t1")]
        report = night_queue.run_night_queue(
            tasks,
            time_budget_s=3600.0,
            heartbeat_path=tmp_path / "heartbeat.json",
            now_fn=clock,
            now_iso_fn=lambda: "2026-08-24T00:00:00Z",
        )
        assert set(report.completed) == {"t0", "t1"}

    def test_non_positive_time_budget_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="time_budget_s"):
            night_queue.run_night_queue(
                [_task("t0")], time_budget_s=0.0, heartbeat_path=tmp_path / "hb.json"
            )

    def test_manifests_written_for_ran_tasks(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        manifests_dir = tmp_path / "manifests"
        night_queue.run_night_queue(
            [_task("t0")],
            time_budget_s=3600.0,
            heartbeat_path=tmp_path / "heartbeat.json",
            manifests_dir=manifests_dir,
            now_fn=clock,
            now_iso_fn=lambda: "2026-08-24T00:00:00Z",
        )
        assert (manifests_dir / "t0.json").is_file()

    def test_final_heartbeat_always_written(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        heartbeat_path = tmp_path / "heartbeat.json"
        night_queue.run_night_queue(
            [_task("t0")],
            time_budget_s=3600.0,
            heartbeat_path=heartbeat_path,
            now_fn=clock,
            now_iso_fn=lambda: "2026-08-24T00:00:00Z",
        )
        assert heartbeat_path.is_file()
        payload = json.loads(heartbeat_path.read_text())
        assert payload["task_count"] == 1
