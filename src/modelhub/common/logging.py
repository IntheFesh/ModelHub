"""Structured JSON logging with fixed fields and secret redaction.

CLAUDE.md §8:
  - structured JSON, fixed fields `run_id / stage / component / level`
  - exceptions logged with full traceback, never just `str(e)`
  - sensitive info (API keys, DB passwords, tokens) redacted
  - logs MAY be truncated (with an explicit `…(truncated N chars)` marker);
    DATA must never be silently truncated the same way — the two code paths
    must be kept separate. `truncate_for_log` in this module is for the
    logging path only; never call it on data returned to a caller.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelhub.common.errors import Stage

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|password|passwd|secret|authorization|credential)", re.IGNORECASE
)
_REDACTED = "***REDACTED***"

# Standard attributes every stdlib LogRecord carries. Anything else on the
# record came from an `extra={...}` at the call site and is surfaced under
# a `fields` sub-object rather than mixed into the fixed top-level schema.
_STANDARD_LOGRECORD_KEYS = frozenset(
    logging.LogRecord("x", logging.INFO, "path", 1, "msg", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "modelhub_run_id", default=None
)


def bind_run_id(run_id: str | None) -> None:
    """Set the run_id that `get_logger()` loggers pick up for this context."""
    _current_run_id.set(run_id)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _SENSITIVE_KEY_PATTERN.search(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    return value


def truncate_for_log(text: str, limit: int = 4000) -> str:
    """Truncate a string for LOG output only — never use this on real data.

    CLAUDE.md §8: logs may be truncated with an explicit marker; the data
    itself (e.g. a SQL result set) must instead carry an explicit
    `truncated: bool` field (see sqlexec's ResultSet) rather than being cut
    here. Mixing the two code paths is exactly what CLAUDE.md forbids.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(truncated {len(text) - limit} chars)"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", None) or _current_run_id.get(),
            "stage": getattr(record, "stage", None),
            "component": getattr(record, "component", record.name),
            "message": record.getMessage(),
        }
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_LOGRECORD_KEYS and k not in payload
        }
        if extra_fields:
            payload["fields"] = extra_fields
        if record.exc_info:
            # CLAUDE.md §8: full traceback, never just str(e).
            payload["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(_redact(payload), ensure_ascii=False, default=repr)


class _ContextAdapter(logging.LoggerAdapter[logging.Logger]):
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        base_extra: dict[str, Any] = dict(self.extra) if self.extra else {}
        extra = {**base_extra, **kwargs.get("extra", {})}
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(
    component: str,
    *,
    run_id: str | None = None,
    stage: Stage | None = None,
    stream: Any = None,
) -> logging.LoggerAdapter[logging.Logger]:
    """Get a JSON-structured logger bound to `component` (and optionally `run_id`/`stage`).

    Idempotent: calling this repeatedly for the same component does not
    stack duplicate handlers.
    """
    logger_name = f"modelhub.{component}"
    base = logging.getLogger(logger_name)
    if not any(isinstance(h, logging.StreamHandler) for h in base.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        base.addHandler(handler)
        base.setLevel(logging.INFO)
        base.propagate = False
    ctx = {"component": component, "run_id": run_id, "stage": stage.value if stage else None}
    return _ContextAdapter(base, ctx)


def log_exception(
    logger: logging.LoggerAdapter[logging.Logger], message: str, exc: BaseException, **fields: Any
) -> None:
    """Log an exception with its full traceback attached — never `str(exc)` alone."""
    logger.error(message, exc_info=exc, extra=fields)
