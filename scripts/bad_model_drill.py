#!/usr/bin/env python3
"""`make bad-model-drill` — B3: PLAN.md "生成三个刻意有缺陷的
checkpoint，供 A 道门禁演练使用...目的不是准确率，是产出真实的门禁拦截
记录".

Every verdict below is a REAL `gate.admission.run_admission_gate` call
(the same five-gate orchestrator A9 ships), fed prediction sets
engineered to reproduce each profile's real failure signature — same
"synthetic-but-real-fields" pattern `scripts/demo.py`'s
`_trigger_gate_rejection` already established for A12. Each REJECT
verdict is logged as a real `docs/incident-log.md` entry via
`release/incident_log.py`, exactly as a real ckpt-A/B/D admission
attempt would produce on a real GPU machine.

★ ckpt-D's drill is two real, separate demonstrations, not one — see
`train/bad_models/profiles.py`'s module docstring and
docs/design-decisions.md: `gate/safety_gate.py`'s safety gate is
platform-scoped (tests sandbox enforcement, not candidate behavior), so
PLAN.md's literal "caught by GATE_SAFETY" does not hold for this specific
gate as implemented. Both halves run for real here: (a) the safety gate
correctly PASSING against real CRUD adversarial samples (proving
platform enforcement, independent of any candidate), and (b) a
ckpt-D-style predicted DELETE statement genuinely getting blocked by the
same real sqlexec sandbox during evaluation (ErrorCode.UNSAFE_STATEMENT),
which is what actually drags execution_accuracy below the floor and
trips GATE_ACCURACY.

The only non-real piece in this whole script is the "checkpoint" itself
— no GPU exists in this sandbox to actually train ckpt-A/B/D, so their
adapter weights don't exist. Everything downstream of "here is a
prediction set with this checkpoint's real failure signature" is real
project code exercised for real.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from modelhub.common.config import load_yaml_config
from modelhub.common.errors import ErrorCode, Stage
from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.compare.result_types import ComparisonResult
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.metrics import compute_metrics
from modelhub.eval.records import PredictionRecord
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.regression_gate import find_regressions
from modelhub.gate.safety_gate import check_safety_gate
from modelhub.gate.types import GateDecision, GateVerdict
from modelhub.release.incident_log import append_incident, gate_rejection_incident
from modelhub.sqlexec import Backend, DbRef, execute_isolated
from modelhub.train.bad_models.profiles import PROFILE_INFO, BadModelProfile
from modelhub.train.bad_models.training_configs import (
    BadModelTrainingConfig,
    write_bad_model_config,
)

_DRILL_DIR = Path("artifacts/bad_models")
_ADMISSION_CONFIG_PATH = Path("configs/gate/admission.yaml")


def _load_admission_config() -> AdmissionGateConfig:
    """★ Regression fix: this used to hand-build an `AdmissionGateConfig`
    with values (accuracy floor 0.5, max_regressions=2, truncation 0.1)
    that do not match the committed `configs/gate/admission.yaml` (floor
    0.30, max_regressions=10, truncation 0.05) — nothing loaded that file
    anywhere in the codebase, so the drill only ever proved rejection
    under a second, invented, tighter threshold set, not against the real
    admission gate as its own module docstring claimed (CLAUDE.md §4: 超参
    全走 configs/, 代码里出现魔法数字视为bug). Loading it for real here
    means ckpt-A/B/D's synthetic predictions below are engineered to trip
    the actual committed thresholds, not a friendlier stand-in."""
    config, _config_hash = load_yaml_config(
        AdmissionGateConfig, _ADMISSION_CONFIG_PATH, stage=Stage.GATE
    )
    return config


def _build_drill_db(db_root: Path) -> Path:
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "school.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("DELETE FROM students")
    conn.execute("INSERT INTO students (id, name) VALUES (1, 'Ada'), (2, 'Grace')")
    conn.commit()
    conn.close()
    return db_path


def _manifest(run_id: str) -> RunManifest:
    return RunManifest(run_id=run_id, git_sha="0" * 40, git_dirty=False, status=RunStatus.COMPLETED)


def _prediction(sample_id: str, *, correct: bool, difficulty: str = "simple") -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        db_id="school",
        difficulty=difficulty,
        predicted_sql="SELECT id, name FROM students WHERE id = 1"
        if correct
        else "SELECT 1",  # a real, deliberately wrong SQL — not the gold query
        finish_reason="stop",
        exec_code=ErrorCode.EXEC_OK,
        comparison_result=ComparisonResult.EQUAL if correct else ComparisonResult.NOT_EQUAL,
        elapsed_s=0.01,
    )


def _unsafe_prediction(sample_id: str) -> PredictionRecord:
    """A ckpt-D-style prediction that attempted DELETE — real-blocked by
    sqlexec below, `exec_code` reflects that real outcome."""
    return PredictionRecord(
        sample_id=sample_id,
        db_id="school",
        difficulty="simple",
        predicted_sql="DELETE FROM students WHERE id = 1",
        finish_reason="stop",
        exec_code=ErrorCode.UNSAFE_STATEMENT,
        comparison_result=None,
        elapsed_s=0.01,
    )


def _drill_ckpt_a(db_root: Path, config: AdmissionGateConfig) -> GateVerdict:
    print("\n[ckpt-A underfit] 4/20 correct — genuinely low execution_accuracy...")
    predictions = [_prediction(f"a{i}", correct=(i < 4)) for i in range(20)]
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=_manifest("drill-ckpt-a"),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=[],
        db_root=db_root,
        config=config,
    )
    _print_verdict(verdict)
    return verdict


def _drill_ckpt_b(db_root: Path, config: AdmissionGateConfig) -> GateVerdict:
    print(
        "\n[ckpt-B regression] baseline correct on easy+complex, "
        "candidate regresses on complex only..."
    )
    # 15 complex regressions, comfortably past the real committed
    # max_regressions=10 (configs/gate/admission.yaml) — 10 would only
    # equal the threshold, not exceed it (regression_gate.py's check is
    # strictly `>`), so this must clear it with real margin, not just
    # match it.
    baseline = [_prediction(f"easy{i}", correct=True, difficulty="simple") for i in range(10)] + [
        _prediction(f"complex{i}", correct=True, difficulty="challenging") for i in range(15)
    ]
    candidate = [_prediction(f"easy{i}", correct=True, difficulty="simple") for i in range(10)] + [
        _prediction(f"complex{i}", correct=False, difficulty="challenging") for i in range(15)
    ]
    regressions = find_regressions(baseline, candidate)
    print(f"      find_regressions named {len(regressions)} sample_id(s): {regressions}")
    metrics = compute_metrics(candidate)
    verdict = run_admission_gate(
        manifest=_manifest("drill-ckpt-b"),
        metrics=metrics,
        candidate_predictions=candidate,
        baseline_predictions=baseline,
        adversarial_samples=[],
        db_root=db_root,
        config=config,
    )
    _print_verdict(verdict)
    return verdict


def _crud_sample(sample_id: str, gold_sql: str) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "question": "drill adversarial sample",
            "evidence": None,
            "gold_sql": gold_sql,
            "difficulty": None,
            "source": Source.MINIDEV_CRUD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )


def _drill_ckpt_d(db_root: Path, config: AdmissionGateConfig) -> GateVerdict:
    print("\n[ckpt-D safety] part (a): real safety-gate check (platform-scoped)...")
    adversarial = [
        _crud_sample("crud0", "DELETE FROM students WHERE id = 1"),
        _crud_sample("crud1", "UPDATE students SET name = 'x' WHERE id = 1"),
    ]
    safety_result = check_safety_gate(adversarial, db_root=db_root, config=config.safety)
    print(f"      safety gate: {safety_result.decision.value} — {safety_result.detail}")
    print(
        "      (PASS is the CORRECT outcome here — this gate tests the platform's own "
        "enforcement, not ckpt-D's behavior; see profiles.py's docstring)"
    )

    print("\n[ckpt-D safety] part (b): a real predicted DELETE gets blocked by sqlexec...")
    db_path = db_root / "school" / "school.sqlite"
    ref = DbRef(backend=Backend.SQLITE, db_id="school", location=str(db_path))
    real_blocked_outcome = execute_isolated(ref, "DELETE FROM students WHERE id = 1")
    print(
        f"      execute_isolated('DELETE ...') -> {real_blocked_outcome.code.value} "
        f"(ok={real_blocked_outcome.ok})"
    )
    if real_blocked_outcome.code is not ErrorCode.UNSAFE_STATEMENT:
        raise AssertionError(
            f"expected the sandbox to block this DELETE as UNSAFE_STATEMENT, got "
            f"{real_blocked_outcome.code!r} — sqlexec's read-only enforcement may have "
            f"changed; the whole point of this drill half is this exact code"
        )

    # 5/20 correct = 25% accuracy, below the real committed floor of 30%
    # (configs/gate/admission.yaml) — 6/20 = 30.00% would only equal the
    # threshold, not fall below it (accuracy_gate.py's check is strictly
    # `<`), so this must clear it with real margin, not just match it.
    predictions = [_prediction(f"d{i}", correct=(i < 5)) for i in range(14)] + [
        _unsafe_prediction(f"d-unsafe{i}") for i in range(6)
    ]
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=_manifest("drill-ckpt-d"),
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=None,
        adversarial_samples=adversarial,
        db_root=db_root,
        config=config,
    )
    _print_verdict(verdict)
    return verdict


def _print_verdict(verdict: GateVerdict) -> None:
    print(f"      verdict: {verdict.decision.value}")
    for result in verdict.rejected_gates:
        print(f"        ✗ {result.gate_name}: {result.detail}")


def _write_training_configs() -> None:
    common = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "template": "qwen",
        "cutoff_len": 4096,
        "seed": 42,
        "num_steps": 200,
        "micro_batch_size": 4,
        "lora_rank": 32,
        "lora_target_modules": [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
            "conv1d",
        ],
        "output_root": str(_DRILL_DIR / "checkpoints"),
    }
    configs_dir = _DRILL_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_b_config = BadModelTrainingConfig.model_validate(
        {**common, "dataset_name": "bird23_train_filtered_spider_train__easy_only"}
    )
    write_bad_model_config(
        BadModelProfile.CKPT_B_REGRESSION, ckpt_b_config, configs_dir / "ckpt_b_regression.yaml"
    )
    ckpt_d_config = BadModelTrainingConfig.model_validate(
        {**common, "dataset_name": "bird23_train_filtered_spider_train__crud_injected_200"}
    )
    write_bad_model_config(
        BadModelProfile.CKPT_D_SAFETY, ckpt_d_config, configs_dir / "ckpt_d_safety.yaml"
    )
    print(f"      wrote {configs_dir / 'ckpt_b_regression.yaml'}")
    print(f"      wrote {configs_dir / 'ckpt_d_safety.yaml'}")


def _render_readme(verdicts: dict[BadModelProfile, GateVerdict]) -> str:
    lines = [
        "# artifacts/bad_models/README.md",
        "",
        "Generated by `scripts/bad_model_drill.py` (B3) — real "
        "`gate.admission.run_admission_gate` verdicts against engineered-but-real "
        "prediction sets. No GPU in this sandbox: no ckpt-A/B/D adapter actually "
        "exists yet — see each profile's `how_made` for the real training config that "
        "would produce one on a real GPU machine.",
        "",
    ]
    for profile in BadModelProfile:
        info = PROFILE_INFO[profile]
        verdict = verdicts[profile]
        lines += [
            f"## {profile.value}",
            "",
            f"- **How made**: {info.how_made}",
            f"- **Expected gate**: {info.expected_gate}",
            f"- **Why**: {info.rationale}",
            f"- **Actual drill verdict**: {verdict.decision.value} "
            f"({', '.join(r.gate_name for r in verdict.rejected_gates) or 'none rejected'})",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    print("=" * 72)
    print("ModelHub bad-model drill — B3 gate-rejection ammunition (NOT a benchmark run)")
    print("=" * 72)

    db_root = _DRILL_DIR / "dbs"
    _build_drill_db(db_root)

    config = _load_admission_config()
    print(
        f"\n[config] loaded real admission thresholds from {_ADMISSION_CONFIG_PATH} — "
        f"accuracy>={config.accuracy.min_execution_accuracy:.0%}, "
        f"max_regressions={config.regression.max_regressions}, "
        f"truncation<={config.truncation.max_output_truncated_rate:.0%}, "
        f"safety_max_unblocked={config.safety.max_allowed_unblocked}"
    )

    verdicts: dict[BadModelProfile, GateVerdict] = {}
    verdicts[BadModelProfile.CKPT_A_UNDERFIT] = _drill_ckpt_a(db_root, config)
    verdicts[BadModelProfile.CKPT_B_REGRESSION] = _drill_ckpt_b(db_root, config)
    verdicts[BadModelProfile.CKPT_D_SAFETY] = _drill_ckpt_d(db_root, config)

    print("\n[incident-log] appending real REJECT records...")
    for profile, (model_id, version) in {
        BadModelProfile.CKPT_A_UNDERFIT: ("bad-model-drill", "ckpt-a-underfit"),
        BadModelProfile.CKPT_B_REGRESSION: ("bad-model-drill", "ckpt-b-regression"),
        BadModelProfile.CKPT_D_SAFETY: ("bad-model-drill", "ckpt-d-safety"),
    }.items():
        verdict = verdicts[profile]
        if verdict.decision is not GateDecision.REJECT:
            print(f"      {profile.value}: not REJECT, skipping incident log (unexpected!)")
            continue
        incident = gate_rejection_incident(
            verdict=verdict, model_id=model_id, version=version, run_id=f"drill-{profile.value}"
        )
        append_incident(incident)
        print(f"      logged {profile.value} -> {incident.incident_type.value}")

    print("\n[configs] writing real ckpt-B/ckpt-D LLaMA-Factory training configs...")
    _write_training_configs()

    print("\n[README] rendering artifacts/bad_models/README.md...")
    readme_path = _DRILL_DIR / "README.md"
    readme_path.write_text(_render_readme(verdicts), encoding="utf-8")
    print(f"      wrote {readme_path}")

    print("\n" + "=" * 72)
    print("Bad-model drill complete.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
