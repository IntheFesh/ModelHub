"""Unit tests for gate/pollution_gate.py."""

from __future__ import annotations

from tests.unit.a4.fakes import make_valid_manifest

from modelhub.gate.pollution_gate import check_pollution_gate
from modelhub.gate.types import GateDecision


def test_clean_manifest_passes() -> None:
    manifest = make_valid_manifest(git_dirty=False, contaminated=False)
    result = check_pollution_gate(manifest)
    assert result.decision is GateDecision.PASS


def test_git_dirty_manifest_rejects() -> None:
    manifest = make_valid_manifest(git_dirty=True)
    result = check_pollution_gate(manifest)
    assert result.decision is GateDecision.REJECT
    assert "git_dirty" in result.detail


def test_contaminated_manifest_rejects() -> None:
    manifest = make_valid_manifest(contaminated=True)
    result = check_pollution_gate(manifest)
    assert result.decision is GateDecision.REJECT
