import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelhub.common.errors import ManifestError
from modelhub.common.run_manifest import (
    KernelStatus,
    RunManifest,
    RunStatus,
    git_head_state,
    manifest_path,
    read_manifest,
    write_manifest,
)


def _make_manifest(**overrides: object) -> RunManifest:
    base = {
        "run_id": "20260901-1432-a3f9c1",
        "git_sha": "deadbeef",
        "git_dirty": False,
        "status": RunStatus.COMPLETED,
    }
    base.update(overrides)
    return RunManifest.model_validate(base)


def test_clean_manifest_is_not_polluted() -> None:
    m = _make_manifest()
    assert m.is_polluted is False
    assert m.pollution_reasons() == []


def test_git_dirty_manifest_is_flagged_polluted() -> None:
    m = _make_manifest(git_dirty=True)
    assert m.is_polluted is True
    assert "git_dirty=true" in m.pollution_reasons()


def test_contaminated_manifest_is_flagged_polluted() -> None:
    m = _make_manifest(contaminated=True)
    assert m.is_polluted is True
    assert "contaminated=true" in m.pollution_reasons()


def test_degraded_kernel_status_is_flagged_polluted() -> None:
    m = _make_manifest(kernel_status=KernelStatus(causal_conv1d=False, fla=False, degraded=True))
    assert m.is_polluted is True
    assert "kernel_status.degraded=true" in m.pollution_reasons()


def test_write_then_read_manifest_roundtrip(tmp_path: Path) -> None:
    m = _make_manifest(seed=42, n_samples=500, error_breakdown={"SYNTAX": 3})
    write_manifest(m, artifacts_root=tmp_path)
    path = manifest_path(m.run_id, artifacts_root=tmp_path)
    assert path.exists()
    reloaded = read_manifest(path)
    assert reloaded == m


def test_read_manifest_missing_file_raises_manifest_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        read_manifest(tmp_path / "nope" / "manifest.json")


def test_extra_field_on_manifest_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(
            {
                "run_id": "r1",
                "git_sha": "abc",
                "git_dirty": False,
                "status": "COMPLETED",
                "totally_made_up_field": 1,
            }
        )


def _init_temp_git_repo(repo_dir: Path) -> None:
    def run(*args: str) -> None:
        subprocess.run(args, cwd=repo_dir, check=True, capture_output=True, timeout=10)

    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (repo_dir / "a.txt").write_text("hello")
    run("git", "add", "a.txt")
    run("git", "commit", "-q", "-m", "init")


def test_git_head_state_clean_repo(tmp_path: Path) -> None:
    _init_temp_git_repo(tmp_path)
    sha, dirty = git_head_state(tmp_path)
    assert len(sha) == 40
    assert dirty is False


def test_git_head_state_dirty_repo(tmp_path: Path) -> None:
    _init_temp_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("modified, uncommitted")
    sha, dirty = git_head_state(tmp_path)
    assert dirty is True
    assert len(sha) == 40
