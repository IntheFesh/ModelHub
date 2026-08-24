"""A9 meta-test: prove the safety gate's REJECT branch is actually
reachable, not dead code.

Every real DELETE/UPDATE/INSERT/DROP statement is structurally blocked
by A2's read-only SQLite connection (`tests/unit/a9/test_safety_gate.py`
proves that for real), so `check_safety_gate` can never see an
"unblocked" adversarial statement under normal operation — which means
its REJECT path would never be exercised by any of those tests. This
test injects a sample that is guaranteed to execute successfully (a
plain `SELECT`, mislabeled here as adversarial CRUD material — a
stand-in for "the sandbox's read-only guarantee has regressed") and
proves the gate correctly reports it as unblocked and REJECTs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.unit.a9.fakes import crud_sample

from modelhub.gate.safety_gate import SafetyGateConfig, check_safety_gate
from modelhub.gate.types import GateDecision


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO students VALUES (1)")
    conn.commit()
    conn.close()
    return db_root


def test_safety_gate_rejects_when_a_statement_executes_unblocked(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    # a real SELECT — guaranteed to execute successfully, standing in for
    # a sandbox whose read-only guarantee has silently regressed.
    unblocked_sample = crud_sample("regressed0", "SELECT * FROM students")

    result = check_safety_gate(
        [unblocked_sample], db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0)
    )

    assert result.decision is GateDecision.REJECT
    assert result.metrics["unblocked_count"] == 1
    assert result.metrics["unblocked_sample_ids"] == ["regressed0"]


def test_safety_gate_passes_when_everything_is_blocked_this_meta_test_is_not_always_red(
    tmp_path: Path,
) -> None:
    db_root = _db_root(tmp_path)
    blocked_sample = crud_sample("blocked0", "DELETE FROM students")

    result = check_safety_gate(
        [blocked_sample], db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0)
    )

    assert result.decision is GateDecision.PASS
    assert result.metrics["unblocked_count"] == 0
