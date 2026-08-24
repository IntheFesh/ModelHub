import time
from pathlib import Path

from modelhub.common.errors import ErrorCode
from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.pool import SqlTask, execute_many


def _sqlite_ref(path: Path) -> DbRef:
    return DbRef(backend=Backend.SQLITE, db_id="test_db", location=str(path))


def test_execute_many_returns_one_outcome_per_task(sqlite_db: Path) -> None:
    ref = _sqlite_ref(sqlite_db)
    tasks = [SqlTask(task_id=f"t{i}", ref=ref, sql=f"SELECT {i}") for i in range(10)]
    results = execute_many(tasks, max_concurrency=4)
    assert set(results.keys()) == {f"t{i}" for i in range(10)}
    assert all(o.ok for o in results.values())


def test_execute_many_empty_batch() -> None:
    assert execute_many([]) == {}


def test_one_slow_task_does_not_block_the_rest_of_the_batch(sqlite_db: Path) -> None:
    ref = _sqlite_ref(sqlite_db)
    runaway_sql = (
        "WITH RECURSIVE cnt(x) AS ("
        "  SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 2000000000"
        ") SELECT COUNT(*) FROM cnt"
    )
    tasks = [
        SqlTask(task_id="slow", ref=ref, sql=runaway_sql),
        *[SqlTask(task_id=f"fast{i}", ref=ref, sql="SELECT 1") for i in range(5)],
    ]
    t0 = time.monotonic()
    results = execute_many(tasks, max_concurrency=6, timeout_s=1.5)
    wall_time = time.monotonic() - t0

    assert results["slow"].code is ErrorCode.TIMEOUT
    for i in range(5):
        assert results[f"fast{i}"].ok
    # Bounded by the timeout, not by letting the runaway query run forever.
    assert wall_time < 10.0


def test_mixed_outcomes_are_independently_classified(sqlite_db: Path) -> None:
    ref = _sqlite_ref(sqlite_db)
    tasks = [
        SqlTask(task_id="ok", ref=ref, sql="SELECT 1"),
        SqlTask(task_id="syntax", ref=ref, sql="SELCT 1"),
        SqlTask(task_id="semantic", ref=ref, sql="SELECT * FROM nowhere"),
        SqlTask(task_id="unsafe", ref=ref, sql="DELETE FROM students"),
    ]
    results = execute_many(tasks, max_concurrency=4)
    assert results["ok"].code is ErrorCode.EXEC_OK
    assert results["syntax"].code is ErrorCode.SYNTAX
    assert results["semantic"].code is ErrorCode.SEMANTIC
    assert results["unsafe"].code is ErrorCode.UNSAFE_STATEMENT
