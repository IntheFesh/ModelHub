"""Smoke test: B5's full reward pipeline at shrunk scale — 3 questions
x k=4 rollouts each (PLAN.md's real run is a full GRPO training loop),
driven end to end by real sqlexec execution (concurrent + timed, via
`execute_rollout_sql_batch`) and real A3 comparison against a real gold
result — no hand-assigned exec_code/comparison_result anywhere, unlike
the unit tests which construct `PredictionRecord`s directly to isolate
one function at a time (CLAUDE.md §1.4: every real code path runs; only
rollout count is shrunk)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from modelhub.common.errors import ErrorCode
from modelhub.compare import ComparatorConfig, compare_result_sets
from modelhub.eval.records import PredictionRecord
from modelhub.sqlexec.backends import Backend, DbRef
from modelhub.sqlexec.outcomes import ExecOutcome
from modelhub.sqlexec.pool import SqlTask
from modelhub.sqlexec.sandbox import execute_isolated
from modelhub.train.grpo.group_diagnostics import RolloutGroup, diagnose_group
from modelhub.train.grpo.reward import compute_reward
from modelhub.train.grpo.rollout_timing import RolloutTimingBreakdown, execute_rollout_sql_batch
from modelhub.train.grpo.step_diagnostics import (
    assert_harness_error_rate_ok,
    compute_step_diagnostics,
)

_QUESTIONS = 3
_K = 4
_GOLD_SQL = "SELECT COUNT(*) FROM students"


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO students VALUES (?, ?)", [(1, "ada"), (2, "grace")])
    conn.commit()
    conn.close()
    return db_root


def _predicted_sql_for(question_index: int, rollout_index: int) -> str:
    # a rotating mix of correct / wrong-but-executes / syntax-error
    # candidates so every real code path (CORRECT/INCORRECT reward,
    # non-degenerate groups) gets exercised in this shrunk run.
    cycle = (question_index * _K + rollout_index) % 3
    if cycle == 0:
        return _GOLD_SQL
    if cycle == 1:
        return "SELECT 1"
    return "SELECT * FROM nope"


def _to_prediction(
    task_id: str, outcome: ExecOutcome, gold_outcome: ExecOutcome
) -> PredictionRecord:
    question_id = task_id.split("-r")[0]
    if not outcome.ok:
        return PredictionRecord(
            sample_id=question_id,
            db_id="school",
            difficulty="simple",
            predicted_sql=outcome.sql,
            finish_reason="stop",
            exec_code=outcome.code,
            comparison_result=None,
            elapsed_s=outcome.elapsed_s,
        )
    assert outcome.result is not None and gold_outcome.result is not None
    comparison = compare_result_sets(outcome.result, gold_outcome.result, config=ComparatorConfig())
    return PredictionRecord(
        sample_id=question_id,
        db_id="school",
        difficulty="simple",
        predicted_sql=outcome.sql,
        finish_reason="stop",
        exec_code=ErrorCode.EXEC_OK,
        comparison_result=comparison.result,
        elapsed_s=outcome.elapsed_s,
    )


def test_full_reward_pipeline_three_questions_k4(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    db_path = db_root / "school" / "school.sqlite"
    ref = DbRef(backend=Backend.SQLITE, db_id="school", location=str(db_path))

    gold_outcome = execute_isolated(ref, _GOLD_SQL)
    assert gold_outcome.ok

    rollout_sql_tasks = [
        SqlTask(task_id=f"q{qi}-r{ri}", ref=ref, sql=_predicted_sql_for(qi, ri))
        for qi in range(_QUESTIONS)
        for ri in range(_K)
    ]
    outcomes, sql_wait_s = execute_rollout_sql_batch(
        rollout_sql_tasks, max_concurrency=4, timeout_s=5.0
    )
    assert len(outcomes) == _QUESTIONS * _K
    timing = RolloutTimingBreakdown(total_rollout_s=sql_wait_s + 0.001, sql_wait_s=sql_wait_s)
    assert 0.0 <= timing.sql_wait_share <= 1.0

    all_predictions = [
        _to_prediction(task.task_id, outcomes[task.task_id], gold_outcome)
        for task in rollout_sql_tasks
    ]

    groups = []
    for qi in range(_QUESTIONS):
        question_id = f"q{qi}"
        question_predictions = [p for p in all_predictions if p.sample_id == question_id]
        rewards = tuple(compute_reward(p) for p in question_predictions)
        groups.append(RolloutGroup(question_id=question_id, rewards=rewards))

    group_diagnostics = [diagnose_group(g) for g in groups]
    step_diag = compute_step_diagnostics(1, all_predictions, group_diagnostics)
    assert_harness_error_rate_ok(step_diag)  # must not raise: 0% harness errors here

    assert step_diag.total_rollouts == _QUESTIONS * _K
    assert any(not gd.is_degenerate for gd in group_diagnostics)  # mixed rewards produced variance
