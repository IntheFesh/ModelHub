"""A5 meta-test: an unverified GDN fast-kernel state must make a run's
manifest `is_polluted`, never silently pass through as a clean run.

CLAUDE.md §5.3: "未验证快 kernel 的 run 标 degraded: true" and §3.1: a
`degraded: true` run's numbers must never be cited by any report — the
gate is `RunManifest.is_polluted`/`assert_not_polluted`
(common/run_manifest.py, already covered by its own A0 tests). What this
meta-test actually proves is the *end-to-end wiring*: does A5's real
kernel-detection output, fed into a real manifest, actually trip that
gate? This sandbox has no GPU and no `causal_conv1d`/`fla` installed, so
`detect_gdn_kernel_status()` is exercising a real negative result here,
not an injected/simulated one — the "failure" this test injects is this
sandbox's genuine environment, not a mock.
"""

from __future__ import annotations

from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.serve.kernel_status import detect_gdn_kernel_status


def test_unverified_gdn_kernel_status_pollutes_the_manifest() -> None:
    status = detect_gdn_kernel_status()
    assert status.degraded is True, (
        "test setup assumption broken: this sandbox is expected to have no "
        "GPU / causal_conv1d / fla, so detect_gdn_kernel_status() must come "
        "back degraded for real"
    )

    manifest = RunManifest.model_validate(
        {
            "run_id": "meta-a5-test-run",
            "git_sha": "0" * 40,
            "git_dirty": False,
            "kernel_status": status.model_dump(),
            "status": RunStatus.COMPLETED,
        }
    )

    assert manifest.is_polluted is True
    assert "kernel_status.degraded=true" in manifest.pollution_reasons()


def test_healthy_kernel_status_does_not_pollute_this_meta_test_is_not_always_red() -> None:
    from modelhub.common.run_manifest import KernelStatus

    healthy = KernelStatus(causal_conv1d=True, fla=True, degraded=False)
    manifest = RunManifest.model_validate(
        {
            "run_id": "meta-a5-test-run-healthy",
            "git_sha": "0" * 40,
            "git_dirty": False,
            "kernel_status": healthy.model_dump(),
            "status": RunStatus.COMPLETED,
        }
    )
    assert manifest.is_polluted is False
