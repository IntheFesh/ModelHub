"""Shared pytest fixtures/config.

`scripts/` holds standalone CLI tools (check_no_cheating.py,
check_placeholders.py, night_queue.py, ...) that are not part of the
`modelhub` package. Tests that exercise them import by module name, so put
`scripts/` on `sys.path` once, here, instead of every test file doing it.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── shared fixtures for tests/unit/a2, tests/meta/a2, and anything later
# (compare/, eval/, train/ reward functions) that needs a real SQLite/PG
# database to execute against ────────────────────────────────────────────


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    """A small BIRD-shaped SQLite database, built fresh per test."""
    path = tmp_path / "bird_like.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)")
    conn.executemany(
        "INSERT INTO students (id, name, gpa) VALUES (?, ?, ?)",
        [(i, f"student_{i}", 2.0 + (i % 20) / 10) for i in range(1, 51)],
    )
    conn.commit()
    conn.close()
    return path


TEST_PG_DSN = (
    "host=127.0.0.1 dbname=modelhub_test user=modelhub_test "
    "password=modelhub_test connect_timeout=2"
)


def _postgres_reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect(TEST_PG_DSN):
            return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_reachable(), reason="local PostgreSQL test server not reachable"
)
