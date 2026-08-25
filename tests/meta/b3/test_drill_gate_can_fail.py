"""CLAUDE.md §1.5 required meta-test: proves B3's drill mechanism
(`scripts/bad_model_drill.py`'s three profile functions, run through the
real `run_admission_gate`) genuinely produces REJECT for each bad
profile's real failure signature, and is not permanently red — a
healthy candidate run through the exact same real gate PASSES.

★ Each REJECT is checked against the SPECIFIC gate PLAN.md/profiles.py
claims causes it, not just "some gate rejected" — a bug that swapped
which gate fires (e.g. ckpt-B accidentally caught by accuracy instead of
regression) must turn this file red. This closes a real gap this
verification pass found: the previous version of this file only checked
`verdict.decision is GateDecision.REJECT`, which a wrong-gate-fires
mutation would have silently passed through.

Also proves the drill is checked against the REAL committed admission
thresholds (`configs/gate/admission.yaml`), not a hand-invented,
friendlier stand-in — another real bug this pass found (see
`bad_model_drill._load_admission_config`'s docstring and
docs/design-decisions.md)."""

from __future__ import annotations

from pathlib import Path

import bad_model_drill
from modelhub.gate.admission import run_admission_gate
from modelhub.gate.types import GateDecision


def test_ckpt_a_underfit_signature_really_rejects_via_accuracy(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    config = bad_model_drill._load_admission_config()
    verdict = bad_model_drill._drill_ckpt_a(tmp_path, config)
    assert verdict.decision is GateDecision.REJECT
    result = verdict.result_for("accuracy")
    assert result is not None
    assert result.decision is GateDecision.REJECT


def test_ckpt_b_regression_signature_really_rejects_via_regression(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    config = bad_model_drill._load_admission_config()
    verdict = bad_model_drill._drill_ckpt_b(tmp_path, config)
    assert verdict.decision is GateDecision.REJECT
    result = verdict.result_for("regression")
    assert result is not None
    assert result.decision is GateDecision.REJECT


def test_ckpt_d_safety_signature_really_rejects_via_accuracy_not_safety(tmp_path: Path) -> None:
    """DD-0034: PLAN.md's literal "caught by GATE_SAFETY" does not hold
    for the safety gate as implemented (platform-scoped, never looks at
    candidate predictions) — the real intercepting gate is accuracy. This
    meta-test asserts BOTH halves so a regression in either direction
    (safety gate wrongly starts rejecting, or accuracy gate stops
    rejecting) goes red."""
    bad_model_drill._build_drill_db(tmp_path)
    config = bad_model_drill._load_admission_config()
    verdict = bad_model_drill._drill_ckpt_d(tmp_path, config)
    assert verdict.decision is GateDecision.REJECT
    safety_result = verdict.result_for("safety")
    accuracy_result = verdict.result_for("accuracy")
    assert safety_result is not None and safety_result.decision is GateDecision.PASS
    assert accuracy_result is not None and accuracy_result.decision is GateDecision.REJECT


def test_a_healthy_candidate_is_not_permanently_rejected(tmp_path: Path) -> None:
    """Proves the drill mechanism itself is not stuck red: a candidate
    with no bad-profile signature at all passes the exact same real gate,
    loaded from the exact same real committed config."""
    bad_model_drill._build_drill_db(tmp_path)
    healthy = [bad_model_drill._prediction(f"h{i}", correct=True) for i in range(20)]
    metrics = bad_model_drill.compute_metrics(healthy)
    config = bad_model_drill._load_admission_config()
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
