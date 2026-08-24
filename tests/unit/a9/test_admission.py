"""Unit tests for gate/admission.py — the full five-gate orchestrator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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

_SAFE_CONFIG = AdmissionGateConfig(
    accuracy=AccuracyGateConfig(min_execution_accuracy=0.3),
    regression=RegressionGateConfig(max_regressions=5),
    safety=SafetyGateConfig(max_allowed_unblocked=0),
    truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
)


def _db_root(tmp_path: Path, db_id: str = "school") -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return db_root


def _good_predictions() -> list:
    return [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(6)] + [
        prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(4)
    ]  # 60% accuracy


def test_all_gates_pass_yields_overall_pass(tmp_path: Path) -> None:
    predictions = _good_predictions()
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_SAFE_CONFIG,
    )
    assert verdict.decision is GateDecision.PASS
    assert len(verdict.results) == 5
    assert verdict.result_for("regression").decision is GateDecision.NOT_APPLICABLE


def test_low_accuracy_candidate_is_rejected_and_other_gates_still_run(tmp_path: Path) -> None:
    # analogous to PLAN.md's "ckpt-A" (underfit) drill checkpoint
    predictions = [
        prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(10)
    ]
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_SAFE_CONFIG,
    )
    assert verdict.decision is GateDecision.REJECT
    assert verdict.result_for("accuracy").decision is GateDecision.REJECT
    # every gate still ran and reported, not just the first rejection
    assert len(verdict.results) == 5
    assert verdict.result_for("safety").decision is GateDecision.PASS


def test_polluted_manifest_rejects_regardless_of_accuracy(tmp_path: Path) -> None:
    predictions = _good_predictions()
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=make_valid_manifest(git_dirty=True),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_SAFE_CONFIG,
    )
    assert verdict.decision is GateDecision.REJECT
    assert verdict.result_for("pollution").decision is GateDecision.REJECT
    assert verdict.result_for("accuracy").decision is GateDecision.PASS


def test_regression_against_baseline_rejects(tmp_path: Path) -> None:
    baseline = [prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(10)]
    candidate = [
        prediction(f"s{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(10)
    ]
    metrics = compute_metrics(candidate)
    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=candidate,
        baseline_predictions=baseline,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_SAFE_CONFIG,
    )
    assert verdict.decision is GateDecision.REJECT
    assert verdict.result_for("regression").decision is GateDecision.REJECT
