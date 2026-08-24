"""B3's three deliberately-bad checkpoint profiles (PLAN.md: "生成三个
刻意有缺陷的 checkpoint，供 A 道门禁演练使用...目的不是准确率，是产出
真实的门禁拦截记录").

★ Honesty note on `CKPT_D_SAFETY`'s expected gate: PLAN.md's own wording
says ckpt-D should be "caught by GATE_SAFETY". `gate/safety_gate.py`'s
`check_safety_gate` is, by its own explicit docstring, scoped to the
*platform's* sandbox enforcement — "it does not even look at what the
model would have generated" — not to a candidate model's own predicted
SQL. A model trained on CRUD-contaminated data that starts emitting
DELETE/UPDATE statements has no dedicated admission-gate check for that
specific failure mode; its unsafe predictions get classified
`UNSAFE_STATEMENT` by the real sandbox during evaluation (same as any
other execution failure) and pull down `execution_accuracy`, which is
what actually trips a gate — `GATE_ACCURACY`, not `GATE_SAFETY`. This is
a real gap in A9's five-gate design (not a B3 bug), documented in
docs/design-decisions.md and reflected honestly here rather than
asserting a gate mapping the real code doesn't produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BadModelProfile(StrEnum):
    CKPT_A_UNDERFIT = "CKPT_A_UNDERFIT"
    CKPT_B_REGRESSION = "CKPT_B_REGRESSION"
    CKPT_D_SAFETY = "CKPT_D_SAFETY"


@dataclass(frozen=True)
class BadModelProfileInfo:
    profile: BadModelProfile
    how_made: str
    expected_gate: str
    rationale: str


PROFILE_INFO: dict[BadModelProfile, BadModelProfileInfo] = {
    BadModelProfile.CKPT_A_UNDERFIT: BadModelProfileInfo(
        profile=BadModelProfile.CKPT_A_UNDERFIT,
        how_made=(
            "No separate training run — directly take B1's main SFT run's earliest "
            "checkpoint (train/bad_models/checkpoint_selection.py::"
            "select_underfit_checkpoint), zero additional GPU cost."
        ),
        expected_gate="accuracy",
        rationale=(
            "An early checkpoint has not converged; execution_accuracy should fall "
            "below AccuracyGateConfig.min_execution_accuracy on its own, independent "
            "of any baseline comparison."
        ),
    ),
    BadModelProfile.CKPT_B_REGRESSION: BadModelProfileInfo(
        profile=BadModelProfile.CKPT_B_REGRESSION,
        how_made=(
            "200-step SFT run on Spider-easy + BIRD-simple samples only "
            "(train/bad_models/dataset_filters.py::filter_easy_only) — the model "
            "never sees complex queries during this drill run."
        ),
        expected_gate="regression",
        rationale=(
            "Restricting training to easy samples should leave complex-query "
            "performance no better than (or worse than) the deployed baseline while "
            "easy-query accuracy stays fine in aggregate — a regression the accuracy "
            "gate alone would not catch, but gate/regression_gate.py::find_regressions "
            "names the specific complex sample_ids that regressed."
        ),
    ),
    BadModelProfile.CKPT_D_SAFETY: BadModelProfileInfo(
        profile=BadModelProfile.CKPT_D_SAFETY,
        how_made=(
            "200-step SFT run with 200 CRUD-containing samples "
            "(train/bad_models/dataset_filters.py::inject_crud_samples) mixed into "
            "the training set."
        ),
        expected_gate="accuracy (corrected from PLAN.md's literal 'safety' — see this "
        "module's docstring and docs/design-decisions.md)",
        rationale=(
            "gate/safety_gate.py's check_safety_gate tests the platform sandbox's own "
            "enforcement, not the candidate's predictions — it has no dependency on "
            "which model is being evaluated. A ckpt-D-style model's DELETE/UPDATE "
            "predictions are real-blocked by sqlexec during evaluation "
            "(ErrorCode.UNSAFE_STATEMENT) and count as wrong answers, which is what "
            "actually depresses execution_accuracy below the admission floor."
        ),
    ),
}

__all__ = ["PROFILE_INFO", "BadModelProfile", "BadModelProfileInfo"]
