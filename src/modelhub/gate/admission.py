"""The five-gate admission verdict, run together.

Every gate always runs, even after an earlier one has already rejected —
PLAN.md's admission drill (三条已知坏 checkpoint, each expected to be
caught by a *different* gate) only means anything if a run shows every
gate's outcome, not just the first rejection.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.run_manifest import RunManifest
from modelhub.data.schema import NormalizedSample
from modelhub.eval.metrics import EvalMetrics
from modelhub.eval.records import PredictionRecord
from modelhub.gate.accuracy_gate import AccuracyGateConfig, check_accuracy_gate
from modelhub.gate.pollution_gate import check_pollution_gate
from modelhub.gate.regression_gate import RegressionGateConfig, check_regression_gate
from modelhub.gate.safety_gate import SafetyGateConfig, check_safety_gate
from modelhub.gate.truncation_gate import TruncationGateConfig, check_truncation_gate
from modelhub.gate.types import GateVerdict


class AdmissionGateConfig(ModelHubBaseConfig):
    accuracy: AccuracyGateConfig
    regression: RegressionGateConfig
    safety: SafetyGateConfig
    truncation: TruncationGateConfig


def run_admission_gate(
    *,
    manifest: RunManifest,
    metrics: EvalMetrics,
    candidate_predictions: list[PredictionRecord],
    baseline_predictions: list[PredictionRecord] | None,
    adversarial_samples: Sequence[NormalizedSample],
    db_root: Path,
    config: AdmissionGateConfig,
) -> GateVerdict:
    results = (
        check_pollution_gate(manifest),
        check_accuracy_gate(metrics, config.accuracy),
        check_truncation_gate(metrics, config.truncation),
        check_safety_gate(adversarial_samples, db_root=db_root, config=config.safety),
        check_regression_gate(baseline_predictions, candidate_predictions, config.regression),
    )
    return GateVerdict(results)


__all__ = ["AdmissionGateConfig", "run_admission_gate"]
