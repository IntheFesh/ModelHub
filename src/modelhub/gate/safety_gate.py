"""Gate 3/5: safety — the sandbox's read-only guarantee, proven against
Mini-Dev V2's 270 CRUD adversarial set, not assumed.

`data/safety_adversarial.py`'s docstring: "a BI/analytics platform
should reject 270/270 of these outright." This gate executes each
adversarial sample's `gold_sql` (a real DELETE/UPDATE/INSERT/etc.
statement) through the real sandbox (`sqlexec.execute_isolated`) and
requires every one of them to be blocked.

★ This gate validates the *platform's* enforcement, not the candidate
model's behavior — it does not even look at what the model would have
generated. It is bundled into admission anyway because a release whose
sandbox has silently regressed (e.g. a backend swap that lost read-only
enforcement) is exactly as dangerous as shipping a bad model, and this
is the cheapest, most direct way to catch that before promoting anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.schema import NormalizedSample, Source
from modelhub.gate.types import GateDecision, GateResult
from modelhub.sqlexec import Backend, DbRef, execute_isolated

_GATE_NAME = "safety"


class SafetyGateConfig(ModelHubBaseConfig):
    max_allowed_unblocked: int


def check_safety_gate(
    adversarial_samples: Sequence[NormalizedSample], *, db_root: Path, config: SafetyGateConfig
) -> GateResult:
    wrong_source = [s for s in adversarial_samples if s.source is not Source.MINIDEV_CRUD]
    if wrong_source:
        raise ValueError(
            f"{len(wrong_source)} sample(s) passed to the safety gate are not "
            f"Source.MINIDEV_CRUD: {[s.sample_id for s in wrong_source][:10]}"
        )

    if not adversarial_samples:
        # An empty set means this check never actually ran against
        # anything — "0/0 unblocked" must not silently read as "verified
        # clean" (gate/types.py's own GateDecision.NOT_APPLICABLE
        # doctrine, CLAUDE.md §1.3's whitelist rule), the same way
        # regression_gate.py already reports NOT_APPLICABLE rather than a
        # vacuous PASS when there is no baseline to compare against.
        return GateResult(
            _GATE_NAME,
            GateDecision.NOT_APPLICABLE,
            "no adversarial CRUD samples supplied — the sandbox's read-only "
            "enforcement was not actually exercised this run",
            {"unblocked_count": 0, "total": 0},
        )

    unblocked: list[str] = []
    for sample in adversarial_samples:
        db_path = db_root / sample.db_id / f"{sample.db_id}.sqlite"
        ref = DbRef(backend=Backend.SQLITE, db_id=sample.db_id, location=str(db_path))
        outcome = execute_isolated(ref, sample.gold_sql)
        if outcome.ok:
            unblocked.append(sample.sample_id)

    total = len(adversarial_samples)
    if len(unblocked) > config.max_allowed_unblocked:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            f"{len(unblocked)}/{total} adversarial CRUD statements executed "
            f"without being blocked (sandbox read-only enforcement failure): "
            f"{unblocked[:10]}",
            {"unblocked_count": len(unblocked), "total": total, "unblocked_sample_ids": unblocked},
        )
    return GateResult(
        _GATE_NAME,
        GateDecision.PASS,
        f"{len(unblocked)}/{total} adversarial CRUD statements unblocked, "
        f"within the allowed {config.max_allowed_unblocked}",
        {"unblocked_count": len(unblocked), "total": total},
    )


__all__ = ["SafetyGateConfig", "check_safety_gate"]
