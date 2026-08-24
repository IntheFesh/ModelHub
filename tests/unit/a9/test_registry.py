"""Unit tests for release/registry.py, against a real filesystem-backed
registry (tmp_path), not an in-memory stand-in."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.gate.types import GateDecision, GateResult, GateVerdict
from modelhub.release.registry import ModelRegistry, ModelStatus


def _pass_verdict() -> GateVerdict:
    return GateVerdict(
        (
            GateResult("accuracy", GateDecision.PASS, "ok", {"execution_accuracy": 0.7}),
            GateResult("pollution", GateDecision.PASS, "clean", {}),
        )
    )


def _reject_verdict() -> GateVerdict:
    return GateVerdict(
        (
            GateResult("accuracy", GateDecision.REJECT, "too low", {"execution_accuracy": 0.1}),
            GateResult("pollution", GateDecision.PASS, "clean", {}),
        )
    )


def test_register_candidate_creates_a_candidate_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register_candidate("arctic-text2sql-r1-7b", "v1", "run-123")
    assert entry.status is ModelStatus.CANDIDATE
    assert entry.run_id == "run-123"

    reloaded = registry.get("arctic-text2sql-r1-7b", "v1")
    assert reloaded == entry


def test_register_candidate_twice_raises(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_candidate("m", "v1", "run-2")


def test_record_pass_verdict_approves(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    entry = registry.record_gate_verdict("m", "v1", _pass_verdict())
    assert entry.status is ModelStatus.APPROVED
    assert entry.gate_results is not None
    assert len(entry.gate_results) == 2


def test_record_reject_verdict_rejects(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    entry = registry.record_gate_verdict("m", "v1", _reject_verdict())
    assert entry.status is ModelStatus.REJECTED


def test_mark_deployed_requires_approved_status(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    with pytest.raises(ValueError, match="not APPROVED"):
        registry.mark_deployed("m", "v1")


def test_mark_deployed_after_approval(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    registry.record_gate_verdict("m", "v1", _pass_verdict())
    entry = registry.mark_deployed("m", "v1")
    assert entry.status is ModelStatus.DEPLOYED
    assert registry.current_deployed("m").version == "v1"


def test_mark_rolled_back_requires_deployed_status(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    registry.record_gate_verdict("m", "v1", _pass_verdict())
    with pytest.raises(ValueError, match="not DEPLOYED"):
        registry.mark_rolled_back("m", "v1")


def test_rollback_after_deployment(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    registry.record_gate_verdict("m", "v1", _pass_verdict())
    registry.mark_deployed("m", "v1")
    entry = registry.mark_rolled_back("m", "v1", notes="perf regression in prod")
    assert entry.status is ModelStatus.ROLLED_BACK
    assert entry.notes == "perf regression in prod"
    assert registry.current_deployed("m") is None


def test_get_unknown_entry_returns_none(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    assert registry.get("no-such-model", "v1") is None


def test_list_entries_filters_by_status(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m1", "v1", "run-1")
    registry.register_candidate("m2", "v1", "run-2")
    registry.record_gate_verdict("m2", "v1", _pass_verdict())

    candidates = registry.list_entries(status=ModelStatus.CANDIDATE)
    approved = registry.list_entries(status=ModelStatus.APPROVED)
    assert [e.model_id for e in candidates] == ["m1"]
    assert [e.model_id for e in approved] == ["m2"]


def test_list_entries_on_empty_registry_returns_empty_list(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    assert registry.list_entries() == []


def test_current_deployed_none_when_nothing_deployed(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("m", "v1", "run-1")
    assert registry.current_deployed("m") is None
