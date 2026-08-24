"""CLAUDE.md §1.5 required meta-test: `test_gate_can_fail`.

"准确率极低的假模型元数据 → 门禁判 REJECT 且失败码正确." Injects a
candidate with deliberately terrible accuracy — the equivalent of
PLAN.md's ckpt-A drill checkpoint — through the real five-gate
`run_admission_gate` orchestrator and asserts the overall verdict is
REJECT, with the correct gate (accuracy) named as the reason. Also
proves the gate is not permanently red (a healthy candidate passes).
"""

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

_CONFIG = AdmissionGateConfig(
    accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
    regression=RegressionGateConfig(max_regressions=5),
    safety=SafetyGateConfig(max_allowed_unblocked=0),
    truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
)


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return db_root


def test_gate_can_fail_on_a_severely_underfit_candidate(tmp_path: Path) -> None:
    # every prediction wrong: 0% execution accuracy, like an early-SFT
    # underfit checkpoint (PLAN.md's "ckpt-A").
    doomed_predictions = [
        prediction(f"s{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(20)
    ]
    metrics = compute_metrics(doomed_predictions)
    assert metrics.execution_accuracy == 0.0

    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=doomed_predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_CONFIG,
    )

    assert verdict.decision is GateDecision.REJECT
    accuracy_result = verdict.result_for("accuracy")
    assert accuracy_result is not None
    assert accuracy_result.decision is GateDecision.REJECT
    assert "0.00%" in accuracy_result.detail or "0%" in accuracy_result.detail


def test_gate_passes_a_healthy_candidate_this_meta_test_is_not_always_red(tmp_path: Path) -> None:
    healthy_predictions = [
        prediction(f"s{i}", comparison_result=ComparisonResult.EQUAL) for i in range(8)
    ] + [prediction(f"f{i}", comparison_result=ComparisonResult.NOT_EQUAL) for i in range(2)]
    metrics = compute_metrics(healthy_predictions)
    assert metrics.execution_accuracy == 0.8

    verdict = run_admission_gate(
        manifest=make_valid_manifest(),
        metrics=metrics,
        candidate_predictions=healthy_predictions,
        baseline_predictions=None,
        adversarial_samples=[crud_sample("c0", "DELETE FROM students")],
        db_root=_db_root(tmp_path),
        config=_CONFIG,
    )
    assert verdict.decision is GateDecision.PASS
