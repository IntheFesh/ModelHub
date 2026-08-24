"""A9 smoke test: run the five-gate admission verdict and record it into
the model registry — the shape a real release pipeline (A10) would
actually call these two modules in."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.unit.a4.fakes import make_valid_manifest
from tests.unit.a9.fakes import crud_sample, prediction

from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import compute_metrics
from modelhub.gate.accuracy_gate import AccuracyGateConfig
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.regression_gate import RegressionGateConfig
from modelhub.gate.safety_gate import SafetyGateConfig
from modelhub.gate.truncation_gate import TruncationGateConfig
from modelhub.gate.types import GateDecision
from modelhub.release.registry import ModelRegistry, ModelStatus

pytestmark = pytest.mark.smoke


def test_gate_and_registry_pipeline_smoke(tmp_path: Path) -> None:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    predictions = [
        prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(7)
    ] + [prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(3)]
    metrics = compute_metrics(predictions)

    config = AdmissionGateConfig(
        accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
        regression=RegressionGateConfig(max_regressions=5),
        safety=SafetyGateConfig(max_allowed_unblocked=0),
        truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
    )

    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=db_root,
        config=config,
    )
    assert verdict.decision is GateDecision.PASS

    registry = ModelRegistry(tmp_path / "registry")
    registry.register_candidate("smoke-model", "v1", "smoke-run-id")
    entry = registry.record_gate_verdict("smoke-model", "v1", verdict)
    assert entry.status is ModelStatus.APPROVED

    deployed = registry.mark_deployed("smoke-model", "v1")
    assert deployed.status is ModelStatus.DEPLOYED
    assert registry.current_deployed("smoke-model").version == "v1"
