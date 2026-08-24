import time
from pathlib import Path

from modelhub.common.errors import ErrorCode
from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.sandbox import execute_isolated


def _sqlite_ref(path: Path) -> DbRef:
    return DbRef(backend=Backend.SQLITE, db_id="test_db", location=str(path))


def test_cpu_bound_cartesian_product_is_actually_killed_not_just_reported_slow(
    sqlite_db: Path,
) -> None:
    """The A2 acceptance question: sqlite3's own `timeout=` kwarg is only a
    busy/lock-wait timeout and cannot stop a CPU-bound query. Prove our
    process-level kill actually works by giving it a query that would run
    for a very long time and a short wall-clock budget."""
    runaway_sql = (
        "WITH RECURSIVE cnt(x) AS ("
        "  SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 2000000000"
        ") SELECT COUNT(*) FROM cnt"
    )
    t0 = time.monotonic()
    outcome = execute_isolated(_sqlite_ref(sqlite_db), runaway_sql, timeout_s=1.0)
    wall_time = time.monotonic() - t0

    assert outcome.code is ErrorCode.TIMEOUT
    # Killed close to the requested timeout, not left running to completion
    # (which for 2e9 recursive steps would be tens of seconds to minutes).
    assert wall_time < 8.0


def test_memory_limit_is_classified_as_timeout_bucket(sqlite_db: Path) -> None:
    # A pathological in-memory blowup (huge GROUP_CONCAT) should hit our
    # RLIMIT_AS cap and come back classified as TIMEOUT (CLAUDE.md: resource
    # exhaustion from a runaway query is the model's fault, same bucket as
    # wall-clock timeout — not a harness bug).
    huge_blowup_sql = (
        "WITH RECURSIVE cnt(x) AS ("
        "  SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5000000"
        ") SELECT group_concat(hex(randomblob(2000))) FROM cnt"
    )
    outcome = execute_isolated(
        _sqlite_ref(sqlite_db), huge_blowup_sql, timeout_s=15.0, memory_limit_mb=64
    )
    assert outcome.code in (ErrorCode.TIMEOUT, ErrorCode.HARNESS_INTERNAL)


def test_elapsed_s_is_populated_on_success(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT 1")
    assert outcome.elapsed_s >= 0.0


def test_elapsed_s_is_populated_on_timeout(sqlite_db: Path) -> None:
    runaway_sql = (
        "WITH RECURSIVE cnt(x) AS ("
        "  SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 2000000000"
        ") SELECT COUNT(*) FROM cnt"
    )
    outcome = execute_isolated(_sqlite_ref(sqlite_db), runaway_sql, timeout_s=0.5)
    assert outcome.elapsed_s >= 0.4  # close to the requested budget, not 0
