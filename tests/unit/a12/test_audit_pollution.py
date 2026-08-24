"""Unit tests for scripts/audit_pollution.py."""

from __future__ import annotations

from pathlib import Path

from tests.unit.a4.fakes import make_valid_manifest

import audit_pollution


def _write_run(root: Path, manifest_kwargs: dict[str, object]) -> None:
    run_dir = root / manifest_kwargs["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(make_valid_manifest(**manifest_kwargs).model_dump_json())


def test_empty_artifacts_root_is_clean(tmp_path: Path) -> None:
    report, all_clean = audit_pollution.render_report(tmp_path / "artifacts")
    assert all_clean is True
    assert "nothing to audit" in report


def test_all_clean_runs_pass(tmp_path: Path) -> None:
    _write_run(tmp_path, {"run_id": "run-a"})
    _write_run(tmp_path, {"run_id": "run-b"})
    report, all_clean = audit_pollution.render_report(tmp_path)
    assert all_clean is True
    assert "run-a" in report
    assert "run-b" in report


def test_a_polluted_run_fails_the_audit(tmp_path: Path) -> None:
    _write_run(tmp_path, {"run_id": "run-good"})
    _write_run(tmp_path, {"run_id": "run-bad", "contaminated": True})
    report, all_clean = audit_pollution.render_report(tmp_path)
    assert all_clean is False
    assert "run-bad" in report
    assert "contaminated=true" in report
