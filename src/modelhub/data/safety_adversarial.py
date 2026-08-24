"""Export Mini-Dev V2's 270 CRUD samples as the safety gate's adversarial set.

CLAUDE.md/A1: these must be kept OUT of the quick-eval set (they'd make the
safety gate look like it's "blocking a correct answer") and instead become
free adversarial test material for A9's safety gate — a BI/analytics
platform should reject 270/270 of these (DELETE/UPDATE/INSERT/ALTER/etc.)
outright.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from modelhub.common.atomic_io import atomic_write_text
from modelhub.data.schema import NormalizedSample, Source


def export_safety_adversarial_set(
    minidev_crud_samples: Sequence[NormalizedSample],
    *,
    out_path: Path = Path("tests/golden/safety_adversarial.jsonl"),
) -> Path:
    wrong_source = [s for s in minidev_crud_samples if s.source is not Source.MINIDEV_CRUD]
    if wrong_source:
        raise ValueError(
            f"{len(wrong_source)} sample(s) are not Source.MINIDEV_CRUD: "
            f"{[s.sample_id for s in wrong_source][:10]}"
        )
    ordered = sorted(minidev_crud_samples, key=lambda s: s.sample_id)
    lines = [
        json.dumps(
            {
                "sample_id": s.sample_id,
                "db_id": s.db_id,
                "question": s.question,
                "sql": s.gold_sql,
                "dialect": s.dialect.value,
                "expected_gate_outcome": "REJECT",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for s in ordered
    ]
    atomic_write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))
    return out_path
