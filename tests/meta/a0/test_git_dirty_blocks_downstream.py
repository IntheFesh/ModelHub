"""A0 acceptance bullet: "git dirty 时 manifest 正确标记且下游引用失败".

Proves `assert_not_polluted` actually goes red for each of the three
pollution flags, not just that `is_polluted` computes correctly (that's
covered in tests/unit/a0/test_run_manifest.py already) — this is the
downstream enforcement point every report generator / gate / docs
renderer is required to call before reusing a run's numbers.
"""

import pytest

from modelhub.common.errors import ErrorCode, ManifestError
from modelhub.common.run_manifest import KernelStatus, RunManifest, RunStatus, assert_not_polluted


def _clean_manifest(**overrides: object) -> RunManifest:
    base = {
        "run_id": "run-under-test",
        "git_sha": "cafebabe",
        "git_dirty": False,
        "status": RunStatus.COMPLETED,
    }
    base.update(overrides)
    return RunManifest.model_validate(base)


def test_git_dirty_run_blocks_downstream_use() -> None:
    dirty_run = _clean_manifest(git_dirty=True)
    with pytest.raises(ManifestError) as exc_info:
        assert_not_polluted(dirty_run, purpose="résumé report")
    assert exc_info.value.code is ErrorCode.RUN_POLLUTED
    assert "git_dirty" in str(exc_info.value)


def test_contaminated_run_blocks_downstream_use() -> None:
    contaminated_run = _clean_manifest(contaminated=True)
    with pytest.raises(ManifestError):
        assert_not_polluted(contaminated_run, purpose="capacity plan")


def test_degraded_kernel_run_blocks_downstream_use() -> None:
    degraded_run = _clean_manifest(
        kernel_status=KernelStatus(causal_conv1d=False, fla=True, degraded=True)
    )
    with pytest.raises(ManifestError):
        assert_not_polluted(degraded_run, purpose="bench report")


def test_clean_run_does_not_block_downstream_use() -> None:
    # Negative case: proves the gate isn't just always-raise.
    clean_run = _clean_manifest()
    assert_not_polluted(clean_run, purpose="résumé report")  # must not raise
