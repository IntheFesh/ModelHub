"""Atomic file writes: write to a sibling ``.tmp`` file, then ``os.replace``.

CLAUDE.md §5.1: "写文件原子写：.tmp → os.replace()。半截文件比没有文件更危险。"
A crash or kill mid-write must never leave a partially-written file at the
target path — readers must see either the previous complete version or the
new complete version, never a truncated one.

This module is the ONLY place in the codebase allowed to write JSON/text
artifacts (manifests, reports, queue snapshots) directly to their final
path; every other module should call through here.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from modelhub.common.errors import ErrorCode, ModelHubError, Stage


def _tmp_path_for(path: Path) -> Path:
    # Same directory as the target so os.replace is an atomic rename on the
    # same filesystem (crossing filesystems would silently fall back to a
    # non-atomic copy on some platforms).
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically. Returns ``path``."""
    path = Path(path)
    tmp = _tmp_path_for(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        # Best-effort cleanup of the temp file; the write error below is
        # what actually matters, so a failure here must not mask it.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise ModelHubError(
            f"atomic write failed for {path}",
            code=ErrorCode.IO_ATOMIC_WRITE_FAILED,
            stage=Stage.DATA,
            context={"path": str(path)},
            retryable=True,
            cause=e,
        ) from e
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> Path:
    """Atomically write ``obj`` as JSON. Keys are sorted for reproducible hashes."""
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, default=str)
    return atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
