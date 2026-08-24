"""The evaluation runner: (model endpoint, dataset version, tier) ->
generate -> execute -> compare -> dump.

Supports --resume-from: predictions already present in an existing
predictions.jsonl are not regenerated (CLAUDE.md §6.3's resume
requirement applies to any long-running unattended job, and a full-tier
eval over BIRD dev is exactly that).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.errors import ErrorCode
from modelhub.compare import ComparatorConfig, compare_result_sets, resolve_row_order
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.data.schema import NormalizedSample
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.model_client import ModelClient
from modelhub.eval.records import PredictionRecord
from modelhub.sqlexec import Backend, DbRef, execute_isolated

EvalTier = Literal["quick", "full"]


def _load_existing(predictions_path: Path) -> dict[str, PredictionRecord]:
    if not predictions_path.is_file():
        return {}
    existing: dict[str, PredictionRecord] = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = PredictionRecord.model_validate_json(line)
        existing[record.sample_id] = record
    return existing


def run_one_sample(
    sample: NormalizedSample,
    *,
    model_client: ModelClient,
    prompt: str,
    db_root: Path,
    gold_cache: GoldExecCache,
    temperature: float,
    max_tokens: int,
    generate_timeout_s: float,
    comparator_config: ComparatorConfig | None = None,
) -> PredictionRecord:
    t0 = time.monotonic()
    generation = model_client.generate(
        prompt, temperature=temperature, max_tokens=max_tokens, timeout_s=generate_timeout_s
    )

    if generation.finish_reason == "length":
        # CLAUDE.md §2.4: truncated output must never be judged as a wrong
        # SQL — it's an incomplete one. Never even attempt to execute it.
        return PredictionRecord(
            sample_id=sample.sample_id,
            db_id=sample.db_id,
            difficulty=sample.difficulty,
            predicted_sql=generation.text,
            finish_reason="length",
            exec_code=ErrorCode.OUTPUT_TRUNCATED,
            comparison_result=None,
            elapsed_s=time.monotonic() - t0,
        )

    db_path = db_root / sample.db_id / f"{sample.db_id}.sqlite"
    predicted_ref = DbRef(backend=Backend.SQLITE, db_id=sample.db_id, location=str(db_path))
    predicted_outcome = execute_isolated(predicted_ref, generation.text)

    if not predicted_outcome.ok:
        return PredictionRecord(
            sample_id=sample.sample_id,
            db_id=sample.db_id,
            difficulty=sample.difficulty,
            predicted_sql=generation.text,
            finish_reason="stop",
            exec_code=predicted_outcome.code,
            comparison_result=None,
            elapsed_s=time.monotonic() - t0,
            error_message=predicted_outcome.error_message,
        )

    gold_outcome = gold_cache.get_or_execute(
        db_id=sample.db_id, gold_sql=sample.gold_sql, db_path=db_path
    )
    if not gold_outcome.ok:
        # CLAUDE.md/A1: some gold SQL genuinely doesn't execute (~425/9428
        # in BIRD-train). We cannot judge correctness against a gold
        # result that doesn't exist — UNDECIDABLE, not a model failure.
        return PredictionRecord(
            sample_id=sample.sample_id,
            db_id=sample.db_id,
            difficulty=sample.difficulty,
            predicted_sql=generation.text,
            finish_reason="stop",
            exec_code=ErrorCode.EXEC_OK,
            comparison_result=ComparisonResult.UNDECIDABLE,
            undecidable_reason=UndecidableReason.GOLD_EXEC_FAILED,
            elapsed_s=time.monotonic() - t0,
            error_message=f"gold SQL failed: {gold_outcome.code.value}",
        )

    assert predicted_outcome.result is not None
    assert gold_outcome.result is not None
    config = (comparator_config or ComparatorConfig()).model_copy(
        update={"row_order": resolve_row_order(sample.gold_sql)}
    )
    comparison = compare_result_sets(predicted_outcome.result, gold_outcome.result, config=config)

    return PredictionRecord(
        sample_id=sample.sample_id,
        db_id=sample.db_id,
        difficulty=sample.difficulty,
        predicted_sql=generation.text,
        finish_reason="stop",
        exec_code=ErrorCode.EXEC_OK,
        comparison_result=comparison.result,
        undecidable_reason=comparison.reason,
        elapsed_s=time.monotonic() - t0,
        error_message=comparison.detail,
    )


def run_eval(
    samples: Sequence[NormalizedSample],
    *,
    model_client: ModelClient,
    prompt_builder: Callable[[NormalizedSample], str],
    db_root: Path,
    predictions_path: Path,
    temperature: float = 0.0,
    max_tokens: int = 512,
    generate_timeout_s: float = 60.0,
    gold_cache: GoldExecCache | None = None,
) -> list[PredictionRecord]:
    """Run (or resume) an eval over `samples`, appending to `predictions_path`
    atomically per-sample-batch. Already-recorded sample_ids are skipped.

    Which tier (`quick`/`full`) this is is a property of *which* `samples`
    the caller passed in, not something this function decides — the tier
    label itself belongs on the manifest/report (see report.py), not here.
    """
    gold_cache = gold_cache or GoldExecCache()
    existing = _load_existing(predictions_path)
    records: list[PredictionRecord] = list(existing.values())

    pending = [s for s in samples if s.sample_id not in existing]
    for sample in pending:
        record = run_one_sample(
            sample,
            model_client=model_client,
            prompt=prompt_builder(sample),
            db_root=db_root,
            gold_cache=gold_cache,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_timeout_s=generate_timeout_s,
        )
        records.append(record)
        _append_and_persist(predictions_path, records)

    return records


def _append_and_persist(predictions_path: Path, records: list[PredictionRecord]) -> None:
    # Atomic per full-list rewrite rather than a raw append: guarantees no
    # torn/partial line can ever be left behind by an interrupted write
    # (CLAUDE.md §5.1/§6.3), at the cost of O(n) rewrite per sample — an
    # acceptable trade for eval-scale (hundreds to low thousands of rows),
    # not for training-scale throughput.
    lines = [r.model_dump_json() for r in records]
    atomic_write_text(predictions_path, "\n".join(lines) + ("\n" if lines else ""))


def load_predictions(predictions_path: Path) -> list[PredictionRecord]:
    return list(_load_existing(predictions_path).values())


__all__ = [
    "EvalTier",
    "load_predictions",
    "run_eval",
    "run_one_sample",
]
