"""SQL execution outcome types (CLAUDE.md §2.3 + §2.4).

``ResultSet.truncated`` is the explicit-truncation-flag pattern CLAUDE.md
§1.2 mandates instead of a silent ``rows[:limit]`` slice: a result that hit
the row cap is marked ``truncated=True`` and ``rows_scanned`` stays
``None`` (unknown) rather than being guessed at — computing the true count
would mean a second, potentially expensive query, and CLAUDE.md §1.1 is
explicit that "missing is None, not 0".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from modelhub.common.errors import ErrorCode


@dataclass(frozen=True)
class ResultSet:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    row_limit: int
    rows_scanned: int | None  # None = unknown, never a stand-in for 0


@dataclass(frozen=True)
class ExecOutcome:
    """The outcome of one SQL execution attempt against one database."""

    code: ErrorCode
    sql: str
    db_id: str
    elapsed_s: float
    result: ResultSet | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code is ErrorCode.EXEC_OK

    @classmethod
    def success(
        cls,
        *,
        sql: str,
        db_id: str,
        elapsed_s: float,
        result: ResultSet,
        context: Mapping[str, Any] | None = None,
    ) -> ExecOutcome:
        return cls(
            code=ErrorCode.EXEC_OK,
            sql=sql,
            db_id=db_id,
            elapsed_s=elapsed_s,
            result=result,
            context=dict(context or {}),
        )

    @classmethod
    def failure(
        cls,
        code: ErrorCode,
        *,
        sql: str,
        db_id: str,
        elapsed_s: float,
        error_message: str | None = None,
        error_traceback: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ExecOutcome:
        if code is ErrorCode.EXEC_OK:
            raise ValueError("ExecOutcome.failure() cannot be called with EXEC_OK")
        return cls(
            code=code,
            sql=sql,
            db_id=db_id,
            elapsed_s=elapsed_s,
            error_message=error_message,
            error_traceback=error_traceback,
            context=dict(context or {}),
        )
