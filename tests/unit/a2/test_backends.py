from pathlib import Path

import pytest
from tests.conftest import TEST_PG_DSN, requires_postgres

from modelhub.common.errors import ErrorCode
from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.sandbox import execute_isolated


def _sqlite_ref(path: Path) -> DbRef:
    return DbRef(backend=Backend.SQLITE, db_id="test_db", location=str(path))


def test_sqlite_select_ok(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT id, name FROM students WHERE id = 1")
    assert outcome.ok
    assert outcome.code is ErrorCode.EXEC_OK
    assert outcome.result is not None
    assert outcome.result.rows == [(1, "student_1")]
    assert outcome.result.truncated is False
    assert outcome.result.rows_scanned == 1


def test_sqlite_syntax_error(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELCT * FROM students")
    assert outcome.code is ErrorCode.SYNTAX
    assert outcome.result is None
    assert outcome.error_message is not None


def test_sqlite_semantic_error_missing_table(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT * FROM nonexistent_table")
    assert outcome.code is ErrorCode.SEMANTIC


def test_sqlite_semantic_error_missing_column(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT nonexistent_col FROM students")
    assert outcome.code is ErrorCode.SEMANTIC


def test_sqlite_db_missing_file(tmp_path: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(tmp_path / "does_not_exist.sqlite"), "SELECT 1")
    assert outcome.code is ErrorCode.HARNESS_DB_UNAVAILABLE


def test_sqlite_write_attempt_is_rejected_not_silently_ignored(sqlite_db: Path) -> None:
    # CLAUDE.md A2 acceptance bullet: DROP TABLE must be provably rejected.
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "DROP TABLE students")
    assert outcome.code is ErrorCode.UNSAFE_STATEMENT
    # And the table must genuinely still be there afterward.
    verify = execute_isolated(_sqlite_ref(sqlite_db), "SELECT COUNT(*) FROM students")
    assert verify.ok
    assert verify.result is not None
    assert verify.result.rows[0][0] == 50


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM students",
        "UPDATE students SET gpa = 0",
        "INSERT INTO students (id, name, gpa) VALUES (999, 'x', 1.0)",
        "ALTER TABLE students ADD COLUMN extra TEXT",
    ],
)
def test_sqlite_all_mutating_statements_are_rejected(sqlite_db: Path, sql: str) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), sql)
    assert outcome.code is ErrorCode.UNSAFE_STATEMENT


def test_sqlite_row_limit_truncation_is_explicit_not_silent(sqlite_db: Path) -> None:
    outcome = execute_isolated(
        _sqlite_ref(sqlite_db), "SELECT id FROM students ORDER BY id", row_limit=10
    )
    assert outcome.ok
    assert outcome.result is not None
    assert len(outcome.result.rows) == 10
    assert outcome.result.truncated is True
    assert outcome.result.row_limit == 10
    # Unknown total, not a guessed 0 or a full second COUNT(*) query.
    assert outcome.result.rows_scanned is None


def test_sqlite_no_truncation_when_under_limit(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT id FROM students", row_limit=1000)
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.truncated is False
    assert outcome.result.rows_scanned == 50


# ── DuckDB (native file, no network needed) ────────────────────────────


def test_duckdb_native_select_ok(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "native.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    conn.close()

    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(path))
    outcome = execute_isolated(ref, "SELECT * FROM t ORDER BY id")
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.rows == [(1, "a"), (2, "b")]


def test_duckdb_syntax_error(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "native2.duckdb"
    duckdb.connect(str(path)).close()
    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(path))
    outcome = execute_isolated(ref, "SELECT * FRM nowhere")
    assert outcome.code is ErrorCode.SYNTAX


def test_duckdb_semantic_error(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "native3.duckdb"
    duckdb.connect(str(path)).close()
    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(path))
    outcome = execute_isolated(ref, "SELECT * FROM nonexistent_table")
    assert outcome.code is ErrorCode.SEMANTIC


def test_duckdb_db_missing_file(tmp_path: Path) -> None:
    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(tmp_path / "nope.duckdb"))
    outcome = execute_isolated(ref, "SELECT 1")
    assert outcome.code is ErrorCode.HARNESS_DB_UNAVAILABLE


def test_duckdb_write_rejected_on_readonly(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "native4.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.close()

    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(path))
    outcome = execute_isolated(ref, "INSERT INTO t VALUES (2)")
    assert outcome.code is ErrorCode.UNSAFE_STATEMENT


@pytest.mark.requires_network
def test_duckdb_attach_sqlite_requires_extension_download(sqlite_db: Path) -> None:
    # FACTS.md's stated production plan: DuckDB ATTACHes the BIRD .sqlite
    # files directly. That needs the sqlite_scanner extension, which this
    # sandbox cannot download (no route to extensions.duckdb.org) — so this
    # is the one sqlexec path we can only exercise on a networked machine.
    ref = DbRef(backend=Backend.DUCKDB, db_id="ddb", location=str(sqlite_db), attach_sqlite=True)
    outcome = execute_isolated(ref, "SELECT COUNT(*) FROM students")
    assert outcome.ok


# ── PostgreSQL (real local server, skipped if unreachable) ─────────────


@requires_postgres
def test_postgres_select_ok() -> None:
    import psycopg

    with psycopg.connect(TEST_PG_DSN) as conn:
        conn.autocommit = True
        conn.execute("DROP TABLE IF EXISTS a2_pg_test")
        conn.execute("CREATE TABLE a2_pg_test (id int, name text)")
        conn.execute("INSERT INTO a2_pg_test VALUES (1, 'a'), (2, 'b')")

    ref = DbRef(backend=Backend.POSTGRES, db_id="pg", location=TEST_PG_DSN)
    outcome = execute_isolated(ref, "SELECT * FROM a2_pg_test ORDER BY id")
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.rows == [(1, "a"), (2, "b")]


@requires_postgres
def test_postgres_syntax_error() -> None:
    ref = DbRef(backend=Backend.POSTGRES, db_id="pg", location=TEST_PG_DSN)
    outcome = execute_isolated(ref, "SELCT * FROM a2_pg_test")
    assert outcome.code is ErrorCode.SYNTAX


@requires_postgres
def test_postgres_semantic_error() -> None:
    ref = DbRef(backend=Backend.POSTGRES, db_id="pg", location=TEST_PG_DSN)
    outcome = execute_isolated(ref, "SELECT * FROM totally_missing_table_xyz")
    assert outcome.code is ErrorCode.SEMANTIC


@requires_postgres
def test_postgres_write_rejected() -> None:
    ref = DbRef(backend=Backend.POSTGRES, db_id="pg", location=TEST_PG_DSN)
    outcome = execute_isolated(ref, "DROP TABLE a2_pg_test")
    assert outcome.code is ErrorCode.UNSAFE_STATEMENT


@requires_postgres
def test_postgres_db_unavailable_bad_host() -> None:
    bad_dsn = "host=127.0.0.1 port=1 dbname=x user=x connect_timeout=1"
    ref = DbRef(backend=Backend.POSTGRES, db_id="pg", location=bad_dsn)
    outcome = execute_isolated(ref, "SELECT 1", timeout_s=5.0)
    assert outcome.code is ErrorCode.HARNESS_DB_UNAVAILABLE
