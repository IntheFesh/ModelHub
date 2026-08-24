"""Sandboxed, timeout-bounded, multi-backend SQL execution.

Public API: `DbRef`/`Backend` to name a database, `execute_isolated` for a
single query, `execute_many`/`SqlTask` for a bounded-concurrency batch.
Every execution returns an `ExecOutcome` classified into the CLAUDE.md
§2.3 seven-way split — never a bare exception, never a silently truncated
result.
"""

from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.outcomes import ExecOutcome, ResultSet
from modelhub.sqlexec.pool import SqlTask, execute_many
from modelhub.sqlexec.sandbox import execute_isolated

__all__ = [
    "Backend",
    "DbRef",
    "ExecOutcome",
    "ResultSet",
    "SqlTask",
    "execute_isolated",
    "execute_many",
]
