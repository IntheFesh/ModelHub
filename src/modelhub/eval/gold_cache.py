"""Gold SQL execution result cache: key = (db_id, gold_sql_hash, db_file_hash).

CLAUDE.md §7.1: "gold SQL 执行结果按 (db_id, gold_sql_hash, db_file_hash) 缓存."
Re-executing the same gold SQL against an unchanged DB file on every
eval run (quick-eval alone reruns 500 gold queries per iteration) is pure
waste; the `db_file_hash` component means a corrupted/updated DB file
invalidates the cache automatically rather than silently serving a stale
gold result forever.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.errors import ErrorCode
from modelhub.data.hashing import sql_hash
from modelhub.sqlexec import Backend, DbRef, ExecOutcome, execute_isolated
from modelhub.sqlexec.outcomes import ResultSet

_HASH_CHUNK_SIZE = 1024 * 1024


def db_file_hash(db_path: Path) -> str:
    digest = hashlib.sha256()
    with open(db_path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(db_id: str, gold_sql: str, db_hash: str) -> str:
    raw = f"{db_id}\x1f{sql_hash(gold_sql)}\x1f{db_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class GoldExecCache:
    def __init__(self, cache_dir: Path = Path("artifacts/cache/gold_exec")) -> None:
        self._cache_dir = cache_dir

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get_or_execute(self, *, db_id: str, gold_sql: str, db_path: Path) -> ExecOutcome:
        db_hash = db_file_hash(db_path)
        key = cache_key(db_id, gold_sql, db_hash)
        cached = self._read(key)
        if cached is not None:
            return cached

        outcome = execute_isolated(
            DbRef(backend=Backend.SQLITE, db_id=db_id, location=str(db_path)), gold_sql
        )
        self._write(key, outcome)
        return outcome

    def _read(self, key: str) -> ExecOutcome | None:
        path = self._path(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = None
        if payload["result"] is not None:
            r = payload["result"]
            result = ResultSet(
                columns=r["columns"],
                rows=[tuple(row) for row in r["rows"]],
                truncated=r["truncated"],
                row_limit=r["row_limit"],
                rows_scanned=r["rows_scanned"],
            )
        return ExecOutcome(
            code=ErrorCode(payload["code"]),
            sql=payload["sql"],
            db_id=payload["db_id"],
            elapsed_s=payload["elapsed_s"],
            result=result,
            error_message=payload["error_message"],
            error_traceback=payload["error_traceback"],
            context=payload["context"],
        )

    def _write(self, key: str, outcome: ExecOutcome) -> None:
        payload = {
            "code": outcome.code.value,
            "sql": outcome.sql,
            "db_id": outcome.db_id,
            "elapsed_s": outcome.elapsed_s,
            "result": (
                {
                    "columns": outcome.result.columns,
                    "rows": [list(row) for row in outcome.result.rows],
                    "truncated": outcome.result.truncated,
                    "row_limit": outcome.result.row_limit,
                    "rows_scanned": outcome.result.rows_scanned,
                }
                if outcome.result is not None
                else None
            ),
            "error_message": outcome.error_message,
            "error_traceback": outcome.error_traceback,
            "context": dict(outcome.context),
        }
        atomic_write_text(self._path(key), json.dumps(payload, sort_keys=True, ensure_ascii=False))
