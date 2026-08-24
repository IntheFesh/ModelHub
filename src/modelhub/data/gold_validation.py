"""Gold SQL execution validation.

CLAUDE.md's A1 acceptance bullet: BIRD-train has 425 known gold SQL that
don't execute. Those failures are DATA, not noise to be dropped — every
one gets dumped to `artifacts/data/<ver>/gold_exec_failures.jsonl` with
its failure code, and the count itself is part of the data-quality report,
never silently absorbed into "usable count".
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.schema import NormalizedSample
from modelhub.sqlexec import Backend, DbRef, ExecOutcome
from modelhub.sqlexec.pool import SqlTask, execute_many


class GoldValidationSummary(ModelHubBaseConfig):
    total: int
    exec_ok: int
    failed: int
    failure_breakdown: dict[str, int]


def validate_gold_sql(
    samples: Sequence[NormalizedSample],
    *,
    db_root: Path,
    max_concurrency: int = 8,
    timeout_s: float = 30.0,
) -> tuple[GoldValidationSummary, list[dict[str, str]]]:
    """Execute every sample's gold SQL against its db. Returns (summary, failures).

    `db_root/<db_id>/<db_id>.sqlite` is the expected BIRD/Spider layout
    (each db_id gets its own directory containing a same-named .sqlite file).
    """
    tasks = [
        SqlTask(
            task_id=s.sample_id,
            ref=DbRef(
                backend=Backend.SQLITE,
                db_id=s.db_id,
                location=str(db_root / s.db_id / f"{s.db_id}.sqlite"),
            ),
            sql=s.gold_sql,
        )
        for s in samples
    ]
    outcomes: dict[str, ExecOutcome] = execute_many(
        tasks, max_concurrency=max_concurrency, timeout_s=timeout_s
    )

    failures: list[dict[str, str]] = []
    breakdown: dict[str, int] = {}
    sample_by_id = {s.sample_id: s for s in samples}
    for task_id, outcome in outcomes.items():
        if outcome.ok:
            continue
        sample = sample_by_id[task_id]
        breakdown[outcome.code.value] = breakdown.get(outcome.code.value, 0) + 1
        failures.append(
            {
                "sample_id": sample.sample_id,
                "db_id": sample.db_id,
                "gold_sql": sample.gold_sql,
                "code": outcome.code.value,
                "error_message": outcome.error_message or "",
            }
        )

    summary = GoldValidationSummary(
        total=len(samples),
        exec_ok=len(samples) - len(failures),
        failed=len(failures),
        failure_breakdown=breakdown,
    )
    return summary, failures


def write_gold_exec_failures(
    failures: list[dict[str, str]],
    *,
    dataset_version: str,
    artifacts_root: Path = Path("artifacts/data"),
) -> Path:
    """Atomically write the failure dump. Never called with an empty list
    silently skipped — an empty list still writes an empty (but present)
    file, so "the file is missing" never gets confused with "zero failures".

    `artifacts_root` defaults to the project convention but is overridable
    so tests/callers never depend on the process's current working
    directory for where this lands."""
    import json

    path = artifacts_root / dataset_version / "gold_exec_failures.jsonl"
    lines = [json.dumps(f, sort_keys=True, ensure_ascii=False) for f in failures]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return path
