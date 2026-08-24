"""Append-only incident log — `docs/incident-log.md`.

PLAN.md's A9 round: "拦截记录 append-only 写 docs/incident-log.md."
A10's automatic-rollback events belong in the same log. Neither A9 nor
A10 (both already shipped and tested) actually wrote to this file —
`gate/admission.py::run_admission_gate` and `release/rollback.py::
trigger_rollback` are deliberately pure/side-effect-light (the gate
decision and the registry mutation respectively), so writing a fixed
docs/ path is not something either of them should do internally. This
module is the missing piece a caller invokes alongside a REJECT verdict
or a triggered rollback — `scripts/demo.py` is the first real caller.

No portable OS-level atomic append exists, so `append_incident` follows
the same read-modify-atomic-write-whole-file pattern `eval/runner.py`'s
`_append_and_persist` already uses: read the current file (or start from
a fresh header), concatenate the new entry, atomic_write_text the whole
thing back (CLAUDE.md §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from modelhub.common.atomic_io import atomic_write_text
from modelhub.gate.types import GateDecision, GateVerdict

_DEFAULT_PATH = Path("docs/incident-log.md")
_HEADER = (
    "# ModelHub · Incident Log\n\n"
    "Append-only. Written by `release/incident_log.py` — never hand-edited.\n"
)


class IncidentType(StrEnum):
    GATE_REJECTION = "GATE_REJECTION"
    CANARY_ROLLBACK = "CANARY_ROLLBACK"


@dataclass(frozen=True)
class IncidentRecord:
    incident_type: IncidentType
    model_id: str
    version: str
    run_id: str
    reason: str
    detail: str
    occurred_at: str


def gate_rejection_incident(
    *,
    verdict: GateVerdict,
    model_id: str,
    version: str,
    run_id: str,
    occurred_at: str | None = None,
) -> IncidentRecord:
    if verdict.decision is not GateDecision.REJECT:
        raise ValueError(
            f"gate_rejection_incident requires a REJECTed GateVerdict, got {verdict.decision.value}"
        )
    rejected = verdict.rejected_gates
    return IncidentRecord(
        incident_type=IncidentType.GATE_REJECTION,
        model_id=model_id,
        version=version,
        run_id=run_id,
        reason="; ".join(f"{r.gate_name}: {r.detail}" for r in rejected),
        detail=f"{len(rejected)} gate(s) rejected: {', '.join(r.gate_name for r in rejected)}",
        occurred_at=occurred_at or datetime.now(UTC).isoformat(),
    )


def canary_rollback_incident(
    *, model_id: str, version: str, run_id: str, reason: str, occurred_at: str | None = None
) -> IncidentRecord:
    return IncidentRecord(
        incident_type=IncidentType.CANARY_ROLLBACK,
        model_id=model_id,
        version=version,
        run_id=run_id,
        reason=reason,
        detail=f"canary rollback triggered: {reason}",
        occurred_at=occurred_at or datetime.now(UTC).isoformat(),
    )


def _render_entry(record: IncidentRecord) -> str:
    return (
        f"## {record.occurred_at} · {record.incident_type.value} · "
        f"{record.model_id}@{record.version}\n\n"
        f"- run_id: `{record.run_id}`\n"
        f"- reason: {record.reason}\n"
        f"- detail: {record.detail}\n"
    )


def append_incident(record: IncidentRecord, *, path: Path = _DEFAULT_PATH) -> Path:
    existing = path.read_text(encoding="utf-8") if path.is_file() else _HEADER
    updated = existing.rstrip("\n") + "\n\n---\n\n" + _render_entry(record) + "\n"
    return atomic_write_text(path, updated)


def read_incident_log(path: Path = _DEFAULT_PATH) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def count_incidents(
    path: Path = _DEFAULT_PATH, *, incident_type: IncidentType | None = None
) -> int:
    """Cheap incident count for the A12 delivery checklist ("门禁拦截与
    灰度回滚各几次") — counts `## ... · <type> · ...` entry headers, not a
    structured re-parse of every field."""
    text = read_incident_log(path)
    if not text:
        return 0
    count = 0
    for line in text.splitlines():
        if not line.startswith("## "):
            continue
        if incident_type is None or f" {incident_type.value} " in line:
            count += 1
    return count


__all__ = [
    "IncidentRecord",
    "IncidentType",
    "append_incident",
    "canary_rollback_incident",
    "count_incidents",
    "gate_rejection_incident",
    "read_incident_log",
]
