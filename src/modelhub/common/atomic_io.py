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
import shutil
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


def atomic_replace_dir(source_dir: Path, target_dir: Path) -> Path:
    """Atomically replace `target_dir` with the already-fully-written
    `source_dir` — the multi-file-checkpoint analogue of
    `atomic_write_bytes`'s single-file `.tmp` -> `os.replace` pattern
    (CLAUDE.md §5.1/§6.3: a checkpoint directory left half-written by a
    crash mid-copy is worse than no checkpoint at all — `train/
    checkpoint_state.py` is what actually builds `source_dir` under a
    temp name before calling this).

    `os.replace` cannot rename onto a non-empty existing directory on
    any platform this project targets, so a pre-existing `target_dir` is
    first renamed aside (a plain, always-atomic rename regardless of
    directory contents) and only deleted once the real swap has
    succeeded — a failure partway through never loses the previous,
    still-valid checkpoint.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.is_dir():
        raise ModelHubError(
            f"atomic_replace_dir source is not a directory: {source_dir}",
            code=ErrorCode.IO_ATOMIC_WRITE_FAILED,
            stage=Stage.DATA,
            context={"source_dir": str(source_dir), "target_dir": str(target_dir)},
            retryable=False,
        )

    stale_dir = target_dir.with_name(f".{target_dir.name}.{uuid.uuid4().hex}.stale")
    moved_stale = False
    try:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            os.replace(target_dir, stale_dir)
            moved_stale = True
        os.replace(source_dir, target_dir)
    except OSError as e:
        # best-effort restore of the previous checkpoint so a failed
        # replace never leaves target_dir missing when it existed before.
        if moved_stale and not target_dir.exists():
            with contextlib.suppress(OSError):
                os.replace(stale_dir, target_dir)
                moved_stale = False
        raise ModelHubError(
            f"atomic directory replace failed for {target_dir}",
            code=ErrorCode.IO_ATOMIC_WRITE_FAILED,
            stage=Stage.DATA,
            context={"source_dir": str(source_dir), "target_dir": str(target_dir)},
            retryable=True,
            cause=e,
        ) from e

    if moved_stale:
        shutil.rmtree(stale_dir, ignore_errors=True)
    return target_dir
