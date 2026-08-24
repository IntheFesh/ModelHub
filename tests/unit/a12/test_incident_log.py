"""Unit tests for release/incident_log.py — the append-only
docs/incident-log.md writer PLAN.md's A9 asked for but neither A9 nor
A10 actually wired up (see the module docstring and DD entry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.gate.types import GateDecision, GateResult, GateVerdict
from modelhub.release.incident_log import (
    IncidentRecord,
    IncidentType,
    append_incident,
    canary_rollback_incident,
    count_incidents,
    gate_rejection_incident,
    read_incident_log,
)


def _reject_verdict() -> GateVerdict:
    return GateVerdict(
        (
            GateResult("accuracy", GateDecision.REJECT, "0.10 < 0.50", {"execution_accuracy": 0.1}),
            GateResult("pollution", GateDecision.PASS, "clean", {}),
        )
    )


def _pass_verdict() -> GateVerdict:
    return GateVerdict((GateResult("accuracy", GateDecision.PASS, "ok", {}),))


def _reject_incident(run_id: str) -> IncidentRecord:
    return gate_rejection_incident(
        verdict=_reject_verdict(), model_id="m", version="v1", run_id=run_id
    )


class TestGateRejectionIncident:
    def test_builds_a_record_from_a_reject_verdict(self) -> None:
        record = gate_rejection_incident(
            verdict=_reject_verdict(), model_id="m", version="v1", run_id="run-1"
        )
        assert record.incident_type is IncidentType.GATE_REJECTION
        assert "accuracy" in record.reason
        assert "0.10 < 0.50" in record.reason

    def test_rejects_a_passing_verdict(self) -> None:
        with pytest.raises(ValueError, match="REJECTed"):
            gate_rejection_incident(
                verdict=_pass_verdict(), model_id="m", version="v1", run_id="run-1"
            )


class TestCanaryRollbackIncident:
    def test_builds_a_record(self) -> None:
        record = canary_rollback_incident(
            model_id="m", version="v2", run_id="run-2", reason="error_rate 50% > 10%"
        )
        assert record.incident_type is IncidentType.CANARY_ROLLBACK
        assert record.reason == "error_rate 50% > 10%"


class TestAppendIncident:
    def test_creates_a_new_file_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "incident-log.md"
        append_incident(_reject_incident("run-1"), path=path)
        text = path.read_text(encoding="utf-8")
        assert "# ModelHub" in text
        assert "GATE_REJECTION" in text
        assert "run-1" in text

    def test_appends_to_an_existing_file_without_losing_prior_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "incident-log.md"
        first = _reject_incident("run-1")
        second = canary_rollback_incident(model_id="m", version="v2", run_id="run-2", reason="劣化")
        append_incident(first, path=path)
        append_incident(second, path=path)
        text = read_incident_log(path)
        assert "run-1" in text
        assert "run-2" in text
        assert text.count("## ") == 2

    def test_read_incident_log_on_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        assert read_incident_log(tmp_path / "no-such-file.md") == ""


class TestCountIncidents:
    def test_counts_all_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "incident-log.md"
        append_incident(_reject_incident("r1"), path=path)
        append_incident(
            canary_rollback_incident(model_id="m", version="v2", run_id="r2", reason="x"), path=path
        )
        assert count_incidents(path) == 2

    def test_counts_filtered_by_type(self, tmp_path: Path) -> None:
        path = tmp_path / "incident-log.md"
        append_incident(_reject_incident("r1"), path=path)
        append_incident(
            canary_rollback_incident(model_id="m", version="v2", run_id="r2", reason="x"), path=path
        )
        assert count_incidents(path, incident_type=IncidentType.GATE_REJECTION) == 1
        assert count_incidents(path, incident_type=IncidentType.CANARY_ROLLBACK) == 1

    def test_zero_on_missing_file(self, tmp_path: Path) -> None:
        assert count_incidents(tmp_path / "no-such-file.md") == 0
