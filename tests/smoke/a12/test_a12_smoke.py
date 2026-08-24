"""Smoke test: A12's wrap-up pieces (incident logging, doc rendering,
pollution audit) chained together at reduced scale (CLAUDE.md §1.4 —
only the input scale shrinks; every real code path still runs). The
full subprocess-based `make demo` flow is verified by hand-running it
(see docs/build-log.md) rather than re-run here, to keep this suite
fast and free of subprocess/network flakiness.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.a9.fakes import crud_sample, prediction

import audit_pollution
import render_docs
from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import compute_metrics
from modelhub.gate.accuracy_gate import AccuracyGateConfig
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.regression_gate import RegressionGateConfig
from modelhub.gate.safety_gate import SafetyGateConfig
from modelhub.gate.truncation_gate import TruncationGateConfig
from modelhub.gate.types import GateDecision
from modelhub.release.incident_log import (
    IncidentType,
    append_incident,
    count_incidents,
    gate_rejection_incident,
)


def test_gate_rejection_flows_into_a_countable_incident_log_entry(tmp_path: Path) -> None:
    import sqlite3

    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    doomed = [prediction(f"s{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(5)]
    metrics = compute_metrics(doomed)
    config = AdmissionGateConfig(
        accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
        regression=RegressionGateConfig(max_regressions=5),
        safety=SafetyGateConfig(max_allowed_unblocked=0),
        truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
    )
    manifest = RunManifest(
        run_id="smoke-run", git_sha="0" * 40, git_dirty=False, status=RunStatus.COMPLETED
    )
    verdict = run_admission_gate(
        manifest=manifest,
        metrics=metrics,
        candidate_predictions=doomed,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("adv0", "DELETE FROM students")],
        db_root=db_root,
        config=config,
    )
    assert verdict.decision is GateDecision.REJECT

    incident_log_path = tmp_path / "incident-log.md"
    incident = gate_rejection_incident(
        verdict=verdict, model_id="smoke-model", version="v1", run_id=manifest.run_id
    )
    append_incident(incident, path=incident_log_path)

    assert count_incidents(incident_log_path, incident_type=IncidentType.GATE_REJECTION) == 1


def test_render_docs_and_audit_pollution_agree_on_run_discovery(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "runs"
    docs_root = tmp_path / "docs"

    written = render_docs.render_all(artifacts_root=artifacts_root, docs_root=docs_root)
    assert len(written) == 3

    report, all_clean = audit_pollution.render_report(artifacts_root)
    assert all_clean is True
    assert "no runs found" in report
