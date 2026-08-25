"""Smoke test: B3's full drill sequence — build db, run all three
profiles through the real admission gate, construct real incident
records — end to end in one shared `tmp_path` (CLAUDE.md §1.4).

`make bad-model-drill` itself (the real script's `main()`, which also
appends to the real `docs/incident-log.md` and writes
`artifacts/bad_models/README.md`) is hand-run and verified for real
instead of wrapped here — same precedent as A12's `scripts/demo.py`
(docs/build-log.md): repeatedly appending to a real, committed file on
every test run is an unwanted side effect a smoke test should not cause.
"""

from __future__ import annotations

from pathlib import Path

import bad_model_drill
from modelhub.gate.types import GateDecision
from modelhub.release.incident_log import gate_rejection_incident


def test_full_drill_sequence_end_to_end(tmp_path: Path) -> None:
    bad_model_drill._build_drill_db(tmp_path)
    config = bad_model_drill._load_admission_config()

    verdicts = {
        bad_model_drill.BadModelProfile.CKPT_A_UNDERFIT: bad_model_drill._drill_ckpt_a(
            tmp_path, config
        ),
        bad_model_drill.BadModelProfile.CKPT_B_REGRESSION: bad_model_drill._drill_ckpt_b(
            tmp_path, config
        ),
        bad_model_drill.BadModelProfile.CKPT_D_SAFETY: bad_model_drill._drill_ckpt_d(
            tmp_path, config
        ),
    }

    for profile, verdict in verdicts.items():
        assert verdict.decision is GateDecision.REJECT, f"{profile} did not reject"
        incident = gate_rejection_incident(
            verdict=verdict,
            model_id="smoke-drill",
            version=profile.value.lower(),
            run_id=f"smoke-{profile.value}",
        )
        assert incident.incident_type.value == "GATE_REJECTION"

    readme = bad_model_drill._render_readme(verdicts)
    assert readme.count("REJECT") == 3
    for profile in bad_model_drill.BadModelProfile:
        assert profile.value in readme
