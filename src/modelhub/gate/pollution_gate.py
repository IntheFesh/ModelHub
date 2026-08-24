"""Gate 4/5: manifest pollution — a thin wrapper around
`common/run_manifest.py`'s existing `is_polluted`/`pollution_reasons`.

A `git_dirty`/`contaminated`/`degraded` eval run's numbers must never be
used to admit a model into production (CLAUDE.md §3.1/§5.2/§5.3), the
same rule that already blocks a polluted run from backing an eval report
(A4) or a served model's kernel-status claim (A5). This gate exists so
that rule is enforced at admission time too, not just at report time.
"""

from __future__ import annotations

from modelhub.common.run_manifest import RunManifest
from modelhub.gate.types import GateDecision, GateResult

_GATE_NAME = "pollution"


def check_pollution_gate(manifest: RunManifest) -> GateResult:
    if manifest.is_polluted:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            f"run {manifest.run_id} is polluted: {', '.join(manifest.pollution_reasons())}",
            {"run_id": manifest.run_id, "reasons": manifest.pollution_reasons()},
        )
    return GateResult(
        _GATE_NAME,
        GateDecision.PASS,
        f"run {manifest.run_id} is clean",
        {"run_id": manifest.run_id},
    )


__all__ = ["check_pollution_gate"]
