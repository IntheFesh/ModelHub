"""Model registry: one file-backed, atomically-written record per
(model_id, version) tracking its admission-gate history and deployment
status.

CLAUDE.md §3.1 applies here the same way it applies to a run manifest —
this is itself state that other tooling (A10's canary rollout, a rollback
decision, "what's currently deployed") reads and must trust, so every
write is atomic (`common/atomic_io.py`) and every entry keeps its full
`GateVerdict` history rather than collapsing it to a final bool.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from modelhub.common.atomic_io import atomic_write_json, read_json
from modelhub.common.config import ModelHubBaseConfig
from modelhub.gate.types import GateDecision, GateVerdict


class ModelStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEPLOYED = "DEPLOYED"
    ROLLED_BACK = "ROLLED_BACK"


class GateResultRecord(ModelHubBaseConfig):
    gate_name: str
    decision: str
    detail: str
    metrics: dict[str, Any]


class RegistryEntry(ModelHubBaseConfig):
    model_id: str
    version: str
    run_id: str
    status: ModelStatus
    registered_at: str
    gate_results: list[GateResultRecord] | None = None
    notes: str | None = None


def _serialize_verdict(verdict: GateVerdict) -> list[GateResultRecord]:
    return [
        GateResultRecord(
            gate_name=r.gate_name,
            decision=r.decision.value,
            detail=r.detail,
            metrics=dict(r.metrics),
        )
        for r in verdict.results
    ]


class ModelRegistry:
    def __init__(self, registry_dir: Path = Path("artifacts/registry")) -> None:
        self._registry_dir = registry_dir

    def _entry_path(self, model_id: str, version: str) -> Path:
        return self._registry_dir / model_id / f"{version}.json"

    def register_candidate(self, model_id: str, version: str, run_id: str) -> RegistryEntry:
        existing = self.get(model_id, version)
        if existing is not None:
            raise ValueError(
                f"model_id={model_id!r} version={version!r} is already registered "
                f"(status={existing.status.value}) — re-registering would silently "
                f"discard its gate history"
            )
        entry = RegistryEntry(
            model_id=model_id,
            version=version,
            run_id=run_id,
            status=ModelStatus.CANDIDATE,
            registered_at=datetime.now(UTC).isoformat(),
        )
        self._write(entry)
        return entry

    def record_gate_verdict(
        self, model_id: str, version: str, verdict: GateVerdict
    ) -> RegistryEntry:
        entry = self._require(model_id, version)
        new_status = (
            ModelStatus.APPROVED if verdict.decision is GateDecision.PASS else ModelStatus.REJECTED
        )
        updated = entry.model_copy(
            update={"status": new_status, "gate_results": _serialize_verdict(verdict)}
        )
        self._write(updated)
        return updated

    def mark_deployed(self, model_id: str, version: str) -> RegistryEntry:
        entry = self._require(model_id, version)
        if entry.status is not ModelStatus.APPROVED:
            raise ValueError(
                f"cannot deploy model_id={model_id!r} version={version!r}: status is "
                f"{entry.status.value}, not APPROVED — a model must pass the admission "
                f"gate before it can be deployed"
            )
        updated = entry.model_copy(update={"status": ModelStatus.DEPLOYED})
        self._write(updated)
        return updated

    def mark_rolled_back(
        self, model_id: str, version: str, *, notes: str | None = None
    ) -> RegistryEntry:
        entry = self._require(model_id, version)
        if entry.status is not ModelStatus.DEPLOYED:
            raise ValueError(
                f"cannot roll back model_id={model_id!r} version={version!r}: status is "
                f"{entry.status.value}, not DEPLOYED"
            )
        updated = entry.model_copy(update={"status": ModelStatus.ROLLED_BACK, "notes": notes})
        self._write(updated)
        return updated

    def get(self, model_id: str, version: str) -> RegistryEntry | None:
        path = self._entry_path(model_id, version)
        if not path.is_file():
            return None
        return RegistryEntry.model_validate(read_json(path))

    def _require(self, model_id: str, version: str) -> RegistryEntry:
        entry = self.get(model_id, version)
        if entry is None:
            raise ValueError(f"no registry entry for model_id={model_id!r} version={version!r}")
        return entry

    def list_entries(self, *, status: ModelStatus | None = None) -> list[RegistryEntry]:
        if not self._registry_dir.is_dir():
            return []
        entries = []
        for model_dir in sorted(self._registry_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for entry_path in sorted(model_dir.glob("*.json")):
                entries.append(RegistryEntry.model_validate(read_json(entry_path)))
        if status is not None:
            entries = [e for e in entries if e.status is status]
        return entries

    def current_deployed(self, model_id: str) -> RegistryEntry | None:
        deployed = [
            e for e in self.list_entries(status=ModelStatus.DEPLOYED) if e.model_id == model_id
        ]
        if len(deployed) > 1:
            raise ValueError(
                f"invariant violated: {len(deployed)} DEPLOYED entries found for "
                f"model_id={model_id!r} ({[e.version for e in deployed]}) — exactly one "
                f"deployed version is expected at a time"
            )
        return deployed[0] if deployed else None

    def _write(self, entry: RegistryEntry) -> None:
        path = self._entry_path(entry.model_id, entry.version)
        atomic_write_json(path, entry.model_dump(mode="json"))


__all__ = ["GateResultRecord", "ModelRegistry", "ModelStatus", "RegistryEntry"]
