"""Model admission gate (five checks) before promotion to production.

Public API: `run_admission_gate`/`AdmissionGateConfig` (admission.py, the
orchestrator — every gate always runs, even after an earlier rejection);
`GateDecision`/`GateResult`/`GateVerdict` (types.py, a three-state
outcome — PASS/REJECT/NOT_APPLICABLE, never a bare bool);
`check_accuracy_gate`/`AccuracyGateConfig` (gate 1: absolute accuracy
floor); `check_regression_gate`/`find_regressions`/`RegressionGateConfig`
(gate 2: CLAUDE.md §1.5's mandated regression detector);
`check_safety_gate`/`SafetyGateConfig` (gate 3: sandbox read-only
enforcement proven against the 270-sample CRUD adversarial set);
`check_pollution_gate` (gate 4: manifest cleanliness);
`check_truncation_gate`/`TruncationGateConfig` (gate 5: output-truncation
rate ceiling).
"""

from modelhub.gate.accuracy_gate import AccuracyGateConfig, check_accuracy_gate
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.pollution_gate import check_pollution_gate
from modelhub.gate.regression_gate import (
    RegressionGateConfig,
    check_regression_gate,
    find_regressions,
)
from modelhub.gate.safety_gate import SafetyGateConfig, check_safety_gate
from modelhub.gate.truncation_gate import TruncationGateConfig, check_truncation_gate
from modelhub.gate.types import GateDecision, GateResult, GateVerdict

__all__ = [
    "AccuracyGateConfig",
    "AdmissionGateConfig",
    "GateDecision",
    "GateResult",
    "GateVerdict",
    "RegressionGateConfig",
    "SafetyGateConfig",
    "TruncationGateConfig",
    "check_accuracy_gate",
    "check_pollution_gate",
    "check_regression_gate",
    "check_safety_gate",
    "check_truncation_gate",
    "find_regressions",
    "run_admission_gate",
]
