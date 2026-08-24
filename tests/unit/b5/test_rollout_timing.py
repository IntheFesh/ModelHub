"""Unit tests for train/grpo/rollout_timing.py — real sqlexec execution
against a real SQLite fixture db, only the wall-clock timer is fake for
determinism."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.pool import SqlTask
from modelhub.train.grpo.rollout_timing import RolloutTimingBreakdown, execute_rollout_sql_batch


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO students VALUES (?, ?)", [(1, "ada"), (2, "grace")])
    conn.commit()
    conn.close()
    return db_root


class TestRolloutTimingBreakdown:
    def test_valid_breakdown_computes_share(self) -> None:
        breakdown = RolloutTimingBreakdown(total_rollout_s=10.0, sql_wait_s=3.0)
        assert breakdown.sql_wait_share == pytest.approx(0.3)

    def test_non_positive_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_rollout_s"):
            RolloutTimingBreakdown(total_rollout_s=0.0, sql_wait_s=0.0)

    def test_negative_sql_wait_rejected(self) -> None:
        with pytest.raises(ValueError, match="sql_wait_s"):
            RolloutTimingBreakdown(total_rollout_s=1.0, sql_wait_s=-1.0)

    def test_sql_wait_exceeding_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            RolloutTimingBreakdown(total_rollout_s=1.0, sql_wait_s=2.0)


class TestExecuteRolloutSqlBatch:
    def test_executes_real_sql_concurrently_and_times_it(self, tmp_path: Path) -> None:
        db_root = _db_root(tmp_path)
        db_path = db_root / "school" / "school.sqlite"
        ref = DbRef(backend=Backend.SQLITE, db_id="school", location=str(db_path))
        tasks = [
            SqlTask(task_id=f"t{i}", ref=ref, sql="SELECT COUNT(*) FROM students") for i in range(5)
        ]
        outcomes, elapsed = execute_rollout_sql_batch(tasks, max_concurrency=4, timeout_s=5.0)
        assert len(outcomes) == 5
        assert all(o.ok for o in outcomes.values())
        assert elapsed >= 0.0

    def test_empty_tasks_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero tasks"):
            execute_rollout_sql_batch([], max_concurrency=4, timeout_s=5.0)
