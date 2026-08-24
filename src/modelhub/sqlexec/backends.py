"""Per-backend connect + run + classify.

Read-only is enforced by each database engine itself, not a regex
blocklist on the SQL text: SQLite via ``mode=ro`` on the file URI, DuckDB
via ``read_only=True``, PostgreSQL via
``default_transaction_read_only``. A blocklist can be evaded by SQL text
games (comments, casing, a CTE hiding a write); the engine's own
enforcement cannot be — and CLAUDE.md §A2 requires the DROP TABLE case be
provably rejected, not "usually caught by our regex".

Exception -> ``ErrorCode`` classification prefers each driver's typed
exception hierarchy (DuckDB and psycopg both expose fine-grained
exception classes) over string matching. SQLite's ``sqlite3`` module does
not distinguish syntax from semantic errors at the exception-class level
(both are ``OperationalError``), so that one backend falls back to a
documented, tested message-substring heuristic — see
``classify_sqlite_exception``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Backend(StrEnum):
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    POSTGRES = "postgres"


@dataclass(frozen=True)
class DbRef:
    """Where to find a database, backend-agnostic."""

    backend: Backend
    db_id: str
    # SQLite/DuckDB native: filesystem path. PostgreSQL: a psycopg conninfo string.
    location: str
    # DuckDB only: ATTACH `location` as a SQLite file via the sqlite_scanner
    # extension instead of opening it as a native .duckdb file. Requires
    # network on first use to download the extension — see tests marked
    # requires_network. This is the mode FACTS.md's plan actually calls for
    # in production ("DuckDB 可直接 attach SQLite，成本极低").
    attach_sqlite: bool = False


@dataclass(frozen=True)
class RawRows:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    has_more: bool  # a (row_limit + 1)-th row existed, i.e. the result was truncated


def _fetch_limited(cursor: Any, row_limit: int) -> tuple[list[Any], bool]:
    rows = cursor.fetchmany(row_limit)
    has_more = cursor.fetchone() is not None
    return rows, has_more


# ── SQLite ───────────────────────────────────────────────────────────────

_SQLITE_UNAVAILABLE_MARKERS = (
    "unable to open database file",
    "database is locked",
)
_SQLITE_INTERNAL_MARKERS = ("database disk image is malformed",)
_SQLITE_SEMANTIC_MARKERS = (
    "no such table",
    "no such column",
    "no such function",
    "ambiguous column name",
    "datatype mismatch",
)


def run_sqlite(db_path: str, sql: str, row_limit: int) -> RawRows:
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"sqlite db not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows, has_more = _fetch_limited(cursor, row_limit)
        return RawRows(columns=columns, rows=rows, has_more=has_more)
    finally:
        conn.close()


def classify_sqlite_exception(exc: BaseException) -> tuple[str, str]:
    from modelhub.common.errors import ErrorCode

    msg = str(exc)
    lower = msg.lower()
    if isinstance(exc, FileNotFoundError):
        return ErrorCode.HARNESS_DB_UNAVAILABLE, msg
    if isinstance(exc, sqlite3.OperationalError):
        if "readonly database" in lower:
            return ErrorCode.UNSAFE_STATEMENT, msg
        if "syntax error" in lower:
            return ErrorCode.SYNTAX, msg
        if any(m in lower for m in _SQLITE_SEMANTIC_MARKERS):
            return ErrorCode.SEMANTIC, msg
        if any(m in lower for m in _SQLITE_UNAVAILABLE_MARKERS):
            return ErrorCode.HARNESS_DB_UNAVAILABLE, msg
        if any(m in lower for m in _SQLITE_INTERNAL_MARKERS):
            return ErrorCode.HARNESS_INTERNAL, msg
        return ErrorCode.UNCLASSIFIED, msg
    if isinstance(exc, sqlite3.ProgrammingError):
        return ErrorCode.SYNTAX, msg
    return ErrorCode.UNCLASSIFIED, msg


# ── DuckDB ───────────────────────────────────────────────────────────────


def run_duckdb(ref: DbRef, sql: str, row_limit: int) -> RawRows:
    import duckdb

    if ref.attach_sqlite:
        conn = duckdb.connect(":memory:")
        try:
            conn.execute("INSTALL sqlite; LOAD sqlite;")
            conn.execute(f"ATTACH '{ref.location}' AS sdb (TYPE sqlite, READ_ONLY)")
            conn.execute("USE sdb")
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows, has_more = _fetch_limited(cursor, row_limit)
            return RawRows(columns=columns, rows=rows, has_more=has_more)
        finally:
            conn.close()
    else:
        if not Path(ref.location).is_file():
            raise duckdb.IOException(f"duckdb db not found: {ref.location}")
        conn = duckdb.connect(ref.location, read_only=True)
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows, has_more = _fetch_limited(cursor, row_limit)
            return RawRows(columns=columns, rows=rows, has_more=has_more)
        finally:
            conn.close()


def classify_duckdb_exception(exc: BaseException) -> tuple[str, str]:
    import duckdb

    from modelhub.common.errors import ErrorCode

    msg = str(exc)
    lower = msg.lower()
    if "read-only" in lower or "readonly" in lower:
        return ErrorCode.UNSAFE_STATEMENT, msg
    if isinstance(exc, duckdb.ParserException | duckdb.SyntaxException):
        return ErrorCode.SYNTAX, msg
    if isinstance(
        exc,
        duckdb.CatalogException
        | duckdb.BinderException
        | duckdb.ConversionException
        | duckdb.TypeMismatchException,
    ):
        return ErrorCode.SEMANTIC, msg
    if isinstance(exc, duckdb.IOException | duckdb.ConnectionException):
        return ErrorCode.HARNESS_DB_UNAVAILABLE, msg
    if isinstance(exc, duckdb.OutOfMemoryException):
        # Same failure-mode bucket as a wall-clock TIMEOUT: a runaway query
        # exhausting resources is the model's fault, not the harness's.
        return ErrorCode.TIMEOUT, msg
    if isinstance(exc, duckdb.InternalException | duckdb.FatalException):
        return ErrorCode.HARNESS_INTERNAL, msg
    return ErrorCode.UNCLASSIFIED, msg


# ── PostgreSQL ───────────────────────────────────────────────────────────


def run_postgres(dsn: str, sql: str, row_limit: int) -> RawRows:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        conn.autocommit = True
        conn.execute("SET default_transaction_read_only = on")
        cursor = conn.execute(sql)
        columns = [d.name for d in cursor.description] if cursor.description else []
        rows, has_more = _fetch_limited(cursor, row_limit)
        return RawRows(columns=columns, rows=rows, has_more=has_more)


def classify_postgres_exception(exc: BaseException) -> tuple[str, str]:
    import psycopg
    from psycopg import errors as pge

    from modelhub.common.errors import ErrorCode

    msg = str(exc)
    if isinstance(exc, pge.ReadOnlySqlTransaction):
        return ErrorCode.UNSAFE_STATEMENT, msg
    if isinstance(exc, pge.SyntaxError):
        return ErrorCode.SYNTAX, msg
    if isinstance(
        exc,
        pge.UndefinedTable
        | pge.UndefinedColumn
        | pge.UndefinedFunction
        | pge.DatatypeMismatch
        | pge.AmbiguousColumn
        | pge.AmbiguousFunction,
    ):
        return ErrorCode.SEMANTIC, msg
    if isinstance(exc, psycopg.OperationalError):
        return ErrorCode.HARNESS_DB_UNAVAILABLE, msg
    return ErrorCode.UNCLASSIFIED, msg
