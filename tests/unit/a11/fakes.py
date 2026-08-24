"""Shared builders for bench/experiments/ (A11) tests — CLAUDE.md §1.4:
plain construction helpers around real project types, not mocks standing
in for the logic under test. Re-exports `tests/unit/a4/fakes.py`'s
`FakeModelClient`/`fixed_response_client`/`make_valid_manifest` and
`tests/unit/a8/fakes.py`'s `BenchFakeClient` rather than duplicating
them — every A11 group reuses the same `ModelClient` Protocol A4/A8's
tests already exercise.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split


def select_sample(
    sample_id: str, *, db_id: str = "school", gold_sql: str = "SELECT id FROM students"
) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": "irrelevant for this test",
            "evidence": None,
            "gold_sql": gold_sql,
            "difficulty": "simple",
            "source": Source.MINIDEV_CRUD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )


def build_school_db(tmp_path: Path, *, db_id: str = "school") -> Path:
    """A real, tiny SQLite db with one `students` table — enough for
    `select_sample`'s default gold_sql to actually execute and return
    a real (non-empty) result set."""
    db_root = tmp_path / "dbs"
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO students (id, name) VALUES (1, 'Ada'), (2, 'Grace')")
    conn.commit()
    conn.close()
    return db_root
