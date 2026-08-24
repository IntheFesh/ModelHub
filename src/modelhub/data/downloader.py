"""Real dataset download logic — genuinely untested in this sandbox.

This sandbox has no route to huggingface.co (confirmed: proxy returns 403)
and no verified-current URLs for BIRD/Spider's own hosting, so nothing in
this module has been executed here. It is written to actually work on a
networked machine, not stubbed — CLAUDE.md §1.1 forbids replacing a real
implementation with a fake one just because we can't run it right now.
Every public function here is covered only by `requires_network`-marked
tests (see tests/unit/a1/test_downloader.py), which is the honest way to
say "written, not verified" rather than lying about coverage.

URLs and HF repo ids are NOT hardcoded here as if independently verified —
they are the caller's responsibility (via `configs/data/*.yaml`), because
this session cannot confirm they are current.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from modelhub.common.atomic_io import atomic_write_bytes
from modelhub.common.errors import ErrorCode, ModelHubError, Stage

_MAX_RETRIES = 4
_BACKOFF_BASE_S = 2.0
_CHUNK_SIZE = 1024 * 1024


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    timeout_s: float = 30.0,
    max_retries: int = _MAX_RETRIES,
) -> Path:
    """Stream `url` to `dest` atomically, with bounded retry+backoff on
    retryable failures only (CLAUDE.md §5.1: retrying a 4xx is a bug).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True) as resp:
                if 400 <= resp.status_code < 500:
                    raise ModelHubError(
                        f"download failed with client error {resp.status_code}, not retrying",
                        code=ErrorCode.HARNESS_DB_UNAVAILABLE,
                        stage=Stage.DATA,
                        context={"url": url, "status_code": resp.status_code},
                        retryable=False,
                    )
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                    digest.update(chunk)
                    chunks.append(chunk)
            data = b"".join(chunks)
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ModelHubError(
                    "downloaded file hash mismatch",
                    code=ErrorCode.HARNESS_DB_UNAVAILABLE,
                    stage=Stage.DATA,
                    context={
                        "url": url,
                        "expected_sha256": expected_sha256,
                        "actual_sha256": digest.hexdigest(),
                    },
                    retryable=True,
                )
            atomic_write_bytes(dest, data)
            return dest
        except ModelHubError as e:
            if not e.retryable or attempt == max_retries:
                raise
            last_exc = e
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == max_retries:
                raise ModelHubError(
                    f"download failed after {max_retries + 1} attempts",
                    code=ErrorCode.HARNESS_DB_UNAVAILABLE,
                    stage=Stage.DATA,
                    context={"url": url, "attempts": attempt + 1},
                    retryable=False,
                    cause=e,
                ) from e
            last_exc = e
        time.sleep(_BACKOFF_BASE_S * (2**attempt))
    raise AssertionError(f"unreachable: retry loop exited without returning ({last_exc!r})")


def download_hf_dataset(repo_id: str, *, local_dir: Path, revision: str | None = None) -> Path:
    """Snapshot-download an HF Hub dataset repo (e.g. FACTS.md's
    `birdsql/bird23-train-filtered`, BIRD Mini-Dev V2's dialect variants).
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir), revision=revision
    )
    return Path(path)
