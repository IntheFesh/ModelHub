"""Process-isolated SQL execution with a wall-clock timeout that actually kills.

CLAUDE.md's A2 round explicitly calls out why this has to be process-level,
not a client-side timeout parameter: SQLite's own ``timeout=`` kwarg is
only a *busy* (lock-wait) timeout. It does nothing for a CPU-bound query
like an unconstrained cartesian product, which just keeps burning CPU
inside the C extension. The only reliable way to stop that is to kill the
OS process running it — so every execution runs in its own subprocess,
and a timeout means SIGKILL, not a cooperative cancellation.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue
import resource
import time
import traceback
from typing import Any

from modelhub.common.errors import ErrorCode
from modelhub.sqlexec.backends import (
    Backend,
    DbRef,
    classify_duckdb_exception,
    classify_postgres_exception,
    classify_sqlite_exception,
    run_duckdb,
    run_postgres,
    run_sqlite,
)
from modelhub.sqlexec.outcomes import ExecOutcome, ResultSet

DEFAULT_ROW_LIMIT = 1000
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_LIMIT_MB = 1024

_MP_CONTEXT = mp.get_context("spawn")


def _classify(backend: Backend, exc: BaseException) -> tuple[ErrorCode, str]:
    if backend is Backend.SQLITE:
        code, msg = classify_sqlite_exception(exc)
    elif backend is Backend.DUCKDB:
        code, msg = classify_duckdb_exception(exc)
    elif backend is Backend.POSTGRES:
        code, msg = classify_postgres_exception(exc)
    else:
        raise AssertionError(f"unhandled backend {backend}")
    return ErrorCode(code), msg


def _worker_main(
    ref: DbRef,
    sql: str,
    row_limit: int,
    memory_limit_mb: int,
    result_queue: mp.Queue[dict[str, Any]],
) -> None:
    """Runs inside the child process. Always enqueues a result, never lets
    an exception escape unclassified — the parent has no other way to learn
    what went wrong once this process is killed or exits."""
    # Best-effort: some sandboxes/kernels disallow lowering RLIMIT_AS.
    with contextlib.suppress(ValueError, OSError):
        limit_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    t0 = time.monotonic()
    try:
        if ref.backend is Backend.SQLITE:
            raw = run_sqlite(ref.location, sql, row_limit)
        elif ref.backend is Backend.DUCKDB:
            raw = run_duckdb(ref, sql, row_limit)
        elif ref.backend is Backend.POSTGRES:
            raw = run_postgres(ref.location, sql, row_limit)
        else:
            raise AssertionError(f"unhandled backend {ref.backend}")
        result_queue.put(
            {
                "ok": True,
                "columns": raw.columns,
                "rows": raw.rows,
                "has_more": raw.has_more,
                "elapsed_s": time.monotonic() - t0,
            }
        )
    except MemoryError as e:
        # Same failure-mode bucket as TIMEOUT: a runaway query exhausting
        # resources is the model's fault, not the harness's.
        result_queue.put(
            {
                "ok": False,
                "code": ErrorCode.TIMEOUT.value,
                "message": f"memory limit ({memory_limit_mb} MB) exceeded: {e}",
                "traceback": traceback.format_exc(),
                "elapsed_s": time.monotonic() - t0,
            }
        )
    except Exception as e:
        code, message = _classify(ref.backend, e)
        result_queue.put(
            {
                "ok": False,
                "code": code.value,
                "message": message,
                "traceback": traceback.format_exc(),
                "elapsed_s": time.monotonic() - t0,
            }
        )


def execute_isolated(
    ref: DbRef,
    sql: str,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> ExecOutcome:
    """Execute `sql` against `ref` in an isolated subprocess with a hard
    wall-clock timeout. On timeout the child is SIGKILLed."""
    result_queue: mp.Queue[dict[str, Any]] = _MP_CONTEXT.Queue()
    proc = _MP_CONTEXT.Process(
        target=_worker_main,
        args=(ref, sql, row_limit, memory_limit_mb, result_queue),
        daemon=True,
    )
    t0 = time.monotonic()
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join(5.0)
        return ExecOutcome.failure(
            ErrorCode.TIMEOUT,
            sql=sql,
            db_id=ref.db_id,
            elapsed_s=time.monotonic() - t0,
            error_message=f"query exceeded {timeout_s}s wall-clock timeout and was killed",
            context={"backend": ref.backend.value, "timeout_s": timeout_s},
        )

    elapsed = time.monotonic() - t0
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        # The worker process exited (crash, segfault, killed by the OS OOM
        # killer outside our own RLIMIT_AS) without ever reporting back.
        # This is a harness failure, never the model's fault.
        return ExecOutcome.failure(
            ErrorCode.HARNESS_INTERNAL,
            sql=sql,
            db_id=ref.db_id,
            elapsed_s=elapsed,
            error_message=(
                f"worker process exited (code={proc.exitcode}) without reporting a result"
            ),
            context={"backend": str(ref.backend), "exitcode": proc.exitcode},
        )
    finally:
        proc.join(1.0)

    if payload["ok"]:
        has_more = bool(payload["has_more"])
        rows = list(payload["rows"])
        result = ResultSet(
            columns=list(payload["columns"]),
            rows=rows,
            truncated=has_more,
            row_limit=row_limit,
            rows_scanned=None if has_more else len(rows),
        )
        return ExecOutcome.success(
            sql=sql,
            db_id=ref.db_id,
            elapsed_s=float(payload["elapsed_s"]),
            result=result,
            context={"backend": ref.backend.value},
        )

    return ExecOutcome.failure(
        ErrorCode(payload["code"]),
        sql=sql,
        db_id=ref.db_id,
        elapsed_s=float(payload["elapsed_s"]),
        error_message=str(payload["message"]),
        error_traceback=str(payload["traceback"]),
        context={"backend": ref.backend.value},
    )
