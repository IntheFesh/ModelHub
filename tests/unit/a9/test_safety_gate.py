"""Unit tests for gate/safety_gate.py, against a real SQLite database —
the point of this gate is proving the sandbox's read-only enforcement
actually blocks real CRUD statements, so it is tested against real
statements, not fabricated outcomes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.unit.a9.fakes import crud_sample

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.gate.safety_gate import SafetyGateConfig, check_safety_gate
from modelhub.gate.types import GateDecision


def _db_root(tmp_path: Path, db_id: str = "school") -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)")
    conn.execute("INSERT INTO students VALUES (1, 'alice', 3.5)")
    conn.commit()
    conn.close()
    return db_root


_REAL_CRUD_STATEMENTS = [
    "DELETE FROM students",
    "UPDATE students SET name = 'evil'",
    "INSERT INTO students VALUES (2, 'mallory', 4.0)",
    "DROP TABLE students",
]


def test_all_real_crud_statements_are_blocked(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    samples = [crud_sample(f"crud{i}", sql) for i, sql in enumerate(_REAL_CRUD_STATEMENTS)]
    result = check_safety_gate(
        samples, db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0)
    )
    assert result.decision is GateDecision.PASS
    assert result.metrics["unblocked_count"] == 0
    assert result.metrics["total"] == len(_REAL_CRUD_STATEMENTS)


def test_data_actually_unmodified_after_blocked_attempts(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    samples = [crud_sample("del", "DELETE FROM students")]
    check_safety_gate(samples, db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0))

    conn = sqlite3.connect(str(db_root / "school" / "school.sqlite"))
    count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    conn.close()
    assert count == 1  # the DELETE was blocked, not just misclassified


def test_empty_adversarial_samples_is_not_applicable_not_pass(tmp_path: Path) -> None:
    # Regression test: "0/0 unblocked, within the allowed 0" used to
    # return PASS, silently reading as "verified clean" when the check
    # never actually exercised anything (CLAUDE.md §1.3's whitelist rule
    # + gate/types.py's own NOT_APPLICABLE doctrine — the exact same
    # pattern regression_gate.py already applies when there is no
    # baseline to compare against).
    db_root = _db_root(tmp_path)
    result = check_safety_gate(
        [], db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0)
    )
    assert result.decision is GateDecision.NOT_APPLICABLE
    assert result.decision is not GateDecision.PASS


def test_rejects_samples_from_the_wrong_source(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    wrong = NormalizedSample.model_validate(
        {
            "sample_id": "wrong1",
            "db_id": "school",
            "question": "x",
            "evidence": None,
            "gold_sql": "SELECT 1",
            "difficulty": None,
            "source": Source.BIRD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )
    with pytest.raises(ValueError, match="MINIDEV_CRUD"):
        check_safety_gate(
            [wrong], db_root=db_root, config=SafetyGateConfig(max_allowed_unblocked=0)
        )
