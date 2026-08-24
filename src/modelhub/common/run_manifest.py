"""Run manifest: the one artifact every number-producing execution must write.

CLAUDE.md §3.1: any execution that produces a number must write
``artifacts/runs/<run_id>/manifest.json``. Three pollution flags —
``git_dirty`` / ``contaminated`` / a ``true`` ``kernel_status.degraded`` —
mean the run's numbers must never be quoted by any report or résumé; any
downstream code that reads a manifest to reuse its numbers must hard-fail
on pollution rather than silently proceeding (CLAUDE.md §3.1, §5.2, §5.3).
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from modelhub.common.atomic_io import atomic_write_json, read_json
from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode, ManifestError, Stage

_GIT_TIMEOUT_S = 10.0


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class EngineInfo(ModelHubBaseConfig):
    name: str
    version: str
    commit: str | None = None


class KernelStatus(ModelHubBaseConfig):
    """GDN fast-kernel state (CLAUDE.md §5.3). Only meaningful for Qwen3.5 runs."""

    causal_conv1d: bool
    fla: bool
    degraded: bool


class HardwareInfo(ModelHubBaseConfig):
    gpu: str
    cc: str | None = None
    count: int
    driver: str | None = None
    cuda: str | None = None
    concurrent_procs: int


class RunManifest(ModelHubBaseConfig):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    git_sha: str
    git_dirty: bool

    config_hash: str | None = None
    config_resolved: dict[str, object] | None = None

    dataset_version: str | None = None
    dataset_hash: str | None = None
    eval_tier: Literal["quick", "full"] | None = None

    model_id: str | None = None
    model_sha: str | None = None
    adapter_sha: str | None = None

    comparator_version: str | None = None
    reward_fn_version: str | None = None

    engine: EngineInfo | None = None
    kernel_status: KernelStatus | None = None
    hardware: HardwareInfo | None = None

    measured_peak_tflops: float | None = None
    measured_bw_gbs: float | None = None

    seed: int | None = None
    n_samples: int | None = None
    error_breakdown: dict[str, int] | None = None

    contaminated: bool = False
    status: RunStatus

    @property
    def degraded(self) -> bool:
        return bool(self.kernel_status and self.kernel_status.degraded)

    @property
    def is_polluted(self) -> bool:
        """True if this run's numbers must never be cited by any report."""
        return self.git_dirty or self.contaminated or self.degraded

    def pollution_reasons(self) -> list[str]:
        reasons = []
        if self.git_dirty:
            reasons.append("git_dirty=true")
        if self.contaminated:
            reasons.append("contaminated=true")
        if self.degraded:
            reasons.append("kernel_status.degraded=true")
        return reasons


def git_head_state(repo_dir: Path | None = None) -> tuple[str, bool]:
    """Return (git_sha, git_dirty) for the repo at `repo_dir` (default: cwd)."""
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        sha_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
        status_out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ManifestError(
            "failed to read git HEAD state",
            code=ErrorCode.HARNESS_INTERNAL,
            stage=Stage.DATA,
            context={"repo_dir": str(repo_dir) if repo_dir else "."},
            retryable=False,
            cause=e,
        ) from e
    git_sha = sha_out.stdout.strip()
    git_dirty = bool(status_out.stdout.strip())
    return git_sha, git_dirty


def manifest_path(run_id: str, *, artifacts_root: Path = Path("artifacts/runs")) -> Path:
    return artifacts_root / run_id / "manifest.json"


def write_manifest(manifest: RunManifest, *, artifacts_root: Path = Path("artifacts/runs")) -> Path:
    path = manifest_path(manifest.run_id, artifacts_root=artifacts_root)
    atomic_write_json(path, manifest.model_dump(mode="json"))
    return path


def read_manifest(path: Path) -> RunManifest:
    try:
        raw = read_json(path)
    except (OSError, ValueError) as e:
        raise ManifestError(
            f"failed to read manifest at {path}",
            code=ErrorCode.HARNESS_INTERNAL,
            stage=Stage.DATA,
            context={"path": str(path)},
            retryable=False,
            cause=e,
        ) from e
    return RunManifest.model_validate(raw)


def assert_not_polluted(manifest: RunManifest, *, purpose: str) -> None:
    """Hard-fail if `manifest` is polluted (CLAUDE.md §3.1/§5.2/§5.3).

    Call this at the top of every code path that reuses another run's
    numbers: report generation, gate manifest checks, docs rendering,
    cross-run comparisons. There is no "soft" mode — a polluted run's
    numbers are not numbers this project is allowed to act on.
    """
    if manifest.is_polluted:
        raise ManifestError(
            f"refusing to use run {manifest.run_id} for {purpose}: "
            f"{', '.join(manifest.pollution_reasons())}",
            code=ErrorCode.RUN_POLLUTED,
            stage=Stage.RELEASE,
            context={
                "run_id": manifest.run_id,
                "purpose": purpose,
                "reasons": manifest.pollution_reasons(),
            },
            retryable=False,
        )
