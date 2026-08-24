"""A2 acceptance bullet: "语法错/表不存在/笛卡尔积超时/DB缺失/正常查询/超大结果集，
七类错误码各命中一次".

Six of the seven CLAUDE.md §2.3 codes are genuinely produced by sqlexec
itself: SYNTAX, SEMANTIC, TIMEOUT, EXEC_OK, HARNESS_DB_UNAVAILABLE,
HARNESS_INTERNAL. The seventh, OUTPUT_TRUNCATED, is deliberately NOT
produced here — it means the *model's generation* was cut off by
max_tokens before a complete SQL string ever existed, which sqlexec has
no way to know from the SQL text alone. That classification has to happen
at the layer that talks to the model (A4/A5), before sqlexec is even
called with a (possibly garbage, truncated) SQL string; see
docs/design-decisions.md.
"""

from pathlib import Path

from modelhub.common.errors import ErrorCode
from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.sandbox import execute_isolated


def _sqlite_ref(path: Path) -> DbRef:
    return DbRef(backend=Backend.SQLITE, db_id="test_db", location=str(path))


def test_syntax_error_hits_syntax_code(sqlite_db: Path) -> None:
    assert execute_isolated(_sqlite_ref(sqlite_db), "SELCT 1").code is ErrorCode.SYNTAX


def test_missing_table_hits_semantic_code(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT * FROM ghost_table")
    assert outcome.code is ErrorCode.SEMANTIC


def test_cartesian_product_timeout_hits_timeout_code(sqlite_db: Path) -> None:
    runaway = (
        "WITH RECURSIVE cnt(x) AS ("
        "  SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 2000000000"
        ") SELECT COUNT(*) FROM cnt"
    )
    outcome = execute_isolated(_sqlite_ref(sqlite_db), runaway, timeout_s=1.0)
    assert outcome.code is ErrorCode.TIMEOUT


def test_missing_db_file_hits_harness_db_unavailable_code(tmp_path: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(tmp_path / "nope.sqlite"), "SELECT 1")
    assert outcome.code is ErrorCode.HARNESS_DB_UNAVAILABLE


def test_normal_query_hits_exec_ok_code(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT COUNT(*) FROM students")
    assert outcome.code is ErrorCode.EXEC_OK


def test_oversized_result_is_truncated_not_silently_cut(sqlite_db: Path) -> None:
    outcome = execute_isolated(_sqlite_ref(sqlite_db), "SELECT id FROM students", row_limit=5)
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.truncated is True
    assert len(outcome.result.rows) == 5


def test_worker_crash_hits_harness_internal_code_not_lost(sqlite_db: Path) -> None:
    """A bug in our own dispatch code (unrecognized backend) crashes the
    worker process before it can classify+report anything. Prove the
    parent still surfaces this as HARNESS_INTERNAL — a system bug — rather
    than silently hanging, returning EXEC_OK, or raising an unclassified
    exception into the caller."""
    broken_ref = DbRef(backend="not_a_real_backend", db_id="x", location=str(sqlite_db))  # type: ignore[arg-type]
    outcome = execute_isolated(broken_ref, "SELECT 1", timeout_s=10.0)
    assert outcome.code is ErrorCode.HARNESS_INTERNAL
    assert outcome.error_message is not None
