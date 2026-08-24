"""CLAUDE.md §1.5 required meta-test: proves B3's drill mechanism
(`scripts/bad_model_drill.py`'s three profile functions, run through the
real `run_admission_gate`) genuinely produces REJECT for each bad
profile's real failure signature, and is not permanently red — a
healthy candidate run through the exact same real gate PASSES."""

from __future__ import annotations

from pathlib import Path

import bad_model_drill
from modelhub.gate.accuracy_gate import AccuracyGateConfig
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.regression_gate import RegressionGateConfig
from modelhub.gate.safety_gate import SafetyGateConfig
from modelhub.gate.truncation_gate import TruncationGateConfig
from modelhub.gate.types import GateDecision


def test_ckpt_a_underfit_signature_really_rejects(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    verdict = bad_model_drill._drill_ckpt_a(tmp_path)
    assert verdict.decision is GateDecision.REJECT


def test_ckpt_b_regression_signature_really_rejects(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    verdict = bad_model_drill._drill_ckpt_b(tmp_path)
    assert verdict.decision is GateDecision.REJECT


def test_ckpt_d_safety_signature_really_rejects(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    verdict = bad_model_drill._drill_ckpt_d(tmp_path)
    assert verdict.decision is GateDecision.REJECT


def test_a_healthy_candidate_is_not_permanently_rejected(tmp_path: Path) -> None:
    """Proves the drill mechanism itself is not stuck red: a candidate
    with no bad-profile signature at all passes the same real gate."""
    bad_model_drill._build_drill_db(tmp_path)
    healthy = [bad_model_drill._prediction(f"h{i}", correct=True) for i in range(20)]
    metrics = bad_model_drill.compute_metrics(healthy)
    config = AdmissionGateConfig(
        accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
        regression=RegressionGateConfig(max_regressions=2),
        safety=SafetyGateConfig(max_allowed_unblocked=0),
        truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
    )
    verdict = run_admission_gate(
        manifest=bad_model_drill._manifest("meta-healthy"),
        metrics=metrics,
        candidate_predictions=healthy,
        baseline_predictions=None,
        adversarial_samples=[],
        db_root=tmp_path,
        config=config,
    )
    assert verdict.decision is GateDecision.PASS
