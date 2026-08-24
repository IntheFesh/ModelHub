"""CLAUDE.md §1.5-style meta-test: prove scripts/audit_pollution.py's
pass/fail judgment can actually go red on a real polluted run, and is
not permanently green."""

from __future__ import annotations

from pathlib import Path

from tests.unit.a4.fakes import make_valid_manifest

import audit_pollution


def _write_run(root: Path, **manifest_kwargs: object) -> None:
    run_dir = root / str(manifest_kwargs["run_id"])
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(make_valid_manifest(**manifest_kwargs).model_dump_json())


def test_degraded_kernel_status_fails_the_audit(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="run-degraded",
        kernel_status={"causal_conv1d": True, "fla": False, "degraded": True},
    )
    _, all_clean = audit_pollution.render_report(tmp_path)
    assert all_clean is False


def test_this_audit_is_not_always_red_a_clean_run_passes(tmp_path: Path) -> None:
    _write_run(tmp_path, run_id="run-clean")
    _, all_clean = audit_pollution.render_report(tmp_path)
    assert all_clean is True


def test_main_returns_nonzero_exit_code_on_pollution(tmp_path: Path) -> None:
    _write_run(tmp_path, run_id="run-dirty", git_dirty=True)
    exit_code = audit_pollution.main(["--artifacts-root", str(tmp_path)])
    assert exit_code == 1


def test_main_returns_zero_exit_code_when_clean(tmp_path: Path) -> None:
    _write_run(tmp_path, run_id="run-clean")
    exit_code = audit_pollution.main(["--artifacts-root", str(tmp_path)])
    assert exit_code == 0
