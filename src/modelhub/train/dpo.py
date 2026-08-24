"""B4: DPO on top of B1's SFT LoRA, via TRL's `DPOTrainer` (PLAN.md:
"用 TRL 的 DPOTrainer...不重写训练循环" — same non-negotiable this
project applies everywhere a real library already does the job).

★ This module's real weight is the preference-pair construction
(PLAN.md steps 1-7), not the `DPOTrainer` wiring — that part is 100%
CPU-testable real logic built on A2 (sqlexec) + A3 (compare) + A4's
`eval.runner.run_one_sample` unchanged. `DPOTrainer` construction itself
needs a real loaded model/tokenizer/dataset this sandbox has none of, so
(same discipline as `train/runner.py::run_sft_training`) its only
sandbox-exercised branch is "trl not installed -> raise", never a real
end-to-end run.

Three-tier preference ranking (PLAN.md: "跑通且结果对 ≻ 语法对但结果错
≻ 语法错"):
  - `EXEC_OK_CORRECT`  — `ErrorCode.EXEC_OK` and `ComparisonResult.EQUAL`
  - `EXEC_OK_INCORRECT` — `ErrorCode.EXEC_OK` and not `EQUAL`
  - `EXEC_FAILED` — the model never produced a safely-executable result:
    `SYNTAX`/`SEMANTIC`/`TIMEOUT` (PLAN.md's literal "语法错") plus
    `UNSAFE_STATEMENT` (not named in PLAN.md's original wording, but
    unambiguously the model's own fault and unambiguously worse than
    `EXEC_OK_INCORRECT` — grouped in rather than left unhandled).

Discarded outright, never entering a pair either side (PLAN.md ★★):
  - `HARNESS_ERROR` (`HARNESS_DB_UNAVAILABLE`/`HARNESS_INTERNAL`) — a
    system fault, not the model's fault; pairing on it would teach the
    model to chase noise (CLAUDE.md §2.3).
  - `UNDECIDABLE` (gold SQL itself failed to execute) and
    `OUTPUT_TRUNCATED` — neither is a real correctness signal.

Same-tier pairs are never built ("同级内不配对") — every pair's chosen
side is strictly better-tiered than its rejected side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

from modelhub.common.capability import CheckStatus, is_available
from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult
from modelhub.data.schema import NormalizedSample
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.model_client import ModelClient
from modelhub.eval.records import PredictionRecord
from modelhub.eval.runner import run_one_sample

try:
    from trl import DPOTrainer as _TrlDpoTrainer
except ImportError:
    _TrlDpoTrainer = None

_EXEC_FAILED_CODES = frozenset(
    {ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT, ErrorCode.UNSAFE_STATEMENT}
)


class PreferenceTier(StrEnum):
    EXEC_OK_CORRECT = "EXEC_OK_CORRECT"
    EXEC_OK_INCORRECT = "EXEC_OK_INCORRECT"
    EXEC_FAILED = "EXEC_FAILED"


# Lower rank = strictly preferred. Only cross-tier pairs (rank difference
# != 0) ever become a PreferencePair.
_TIER_RANK: dict[PreferenceTier, int] = {
    PreferenceTier.EXEC_OK_CORRECT: 0,
    PreferenceTier.EXEC_OK_INCORRECT: 1,
    PreferenceTier.EXEC_FAILED: 2,
}


class DiscardReason(StrEnum):
    HARNESS_ERROR = "HARNESS_ERROR"
    UNDECIDABLE = "UNDECIDABLE"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"


@dataclass(frozen=True)
class ClassifiedCandidate:
    question_id: str
    candidate_index: int
    predicted_sql: str
    tier: PreferenceTier | None
    discard_reason: DiscardReason | None

    def __post_init__(self) -> None:
        if (self.tier is None) == (self.discard_reason is None):
            raise ValueError(
                "exactly one of tier/discard_reason must be set, got "
                f"tier={self.tier!r} discard_reason={self.discard_reason!r}"
            )


def classify_prediction(
    question_id: str, candidate_index: int, prediction: PredictionRecord
) -> ClassifiedCandidate:
    """Whitelist-ordered classification — discard checks run first and
    are mutually exclusive with tier assignment by construction
    (`is_harness_error`/`is_output_truncated` can never co-occur with
    `EXEC_OK`, and `UNDECIDABLE` only ever occurs when `exec_code is
    EXEC_OK`, per `eval/runner.py::run_one_sample`)."""

    def _classified(
        *, tier: PreferenceTier | None, discard_reason: DiscardReason | None
    ) -> ClassifiedCandidate:
        return ClassifiedCandidate(
            question_id=question_id,
            candidate_index=candidate_index,
            predicted_sql=prediction.predicted_sql,
            tier=tier,
            discard_reason=discard_reason,
        )

    if prediction.is_harness_error:
        return _classified(tier=None, discard_reason=DiscardReason.HARNESS_ERROR)
    if prediction.is_output_truncated:
        return _classified(tier=None, discard_reason=DiscardReason.OUTPUT_TRUNCATED)
    if prediction.comparison_result is ComparisonResult.UNDECIDABLE:
        return _classified(tier=None, discard_reason=DiscardReason.UNDECIDABLE)
    if prediction.exec_code is ErrorCode.EXEC_OK:
        tier = (
            PreferenceTier.EXEC_OK_CORRECT
            if prediction.comparison_result is ComparisonResult.EQUAL
            else PreferenceTier.EXEC_OK_INCORRECT
        )
        return _classified(tier=tier, discard_reason=None)
    if prediction.exec_code in _EXEC_FAILED_CODES:
        return _classified(tier=PreferenceTier.EXEC_FAILED, discard_reason=None)
    raise ValueError(
        f"unhandled exec_code {prediction.exec_code!r} for {question_id}#{candidate_index} — "
        f"classify_prediction must be exhaustive over every real PredictionRecord.exec_code "
        f"value, not silently bucket an unexpected one (CLAUDE.md §1.3 whitelist rule)"
    )


@dataclass(frozen=True)
class PreferencePair:
    question_id: str
    chosen_sql: str
    rejected_sql: str
    chosen_tier: PreferenceTier
    rejected_tier: PreferenceTier


@dataclass(frozen=True)
class QuestionPreferenceResult:
    question_id: str
    classified: tuple[ClassifiedCandidate, ...]
    pairs: tuple[PreferencePair, ...]


def build_preference_pairs_for_question(
    question_id: str, predictions: Sequence[PredictionRecord]
) -> QuestionPreferenceResult:
    classified = tuple(classify_prediction(question_id, i, p) for i, p in enumerate(predictions))
    kept = [c for c in classified if c.tier is not None]
    pairs = []
    for chosen in kept:
        for rejected in kept:
            if chosen.candidate_index == rejected.candidate_index:
                continue
            assert chosen.tier is not None and rejected.tier is not None
            if _TIER_RANK[chosen.tier] < _TIER_RANK[rejected.tier]:
                pairs.append(
                    PreferencePair(
                        question_id=question_id,
                        chosen_sql=chosen.predicted_sql,
                        rejected_sql=rejected.predicted_sql,
                        chosen_tier=chosen.tier,
                        rejected_tier=rejected.tier,
                    )
                )
    return QuestionPreferenceResult(
        question_id=question_id, classified=classified, pairs=tuple(pairs)
    )


@dataclass(frozen=True)
class PreferenceDatasetReport:
    per_question: tuple[QuestionPreferenceResult, ...]

    @property
    def all_pairs(self) -> tuple[PreferencePair, ...]:
        return tuple(p for q in self.per_question for p in q.pairs)

    @property
    def pair_count(self) -> int:
        return len(self.all_pairs)

    @property
    def total_candidates(self) -> int:
        return sum(len(q.classified) for q in self.per_question)

    @property
    def tier_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {tier.value: 0 for tier in PreferenceTier}
        for q in self.per_question:
            for c in q.classified:
                if c.tier is not None:
                    counts[c.tier.value] += 1
        return counts

    @property
    def discard_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {reason.value: 0 for reason in DiscardReason}
        for q in self.per_question:
            for c in q.classified:
                if c.discard_reason is not None:
                    counts[c.discard_reason.value] += 1
        return counts

    @property
    def exec_ok_correct_share(self) -> float:
        """PLAN.md's specific check: "如果「跑通且对」占比已经很高，
        DPO 收益会很小" — share among KEPT (non-discarded) candidates."""
        tiers = self.tier_counts
        kept_total = sum(tiers.values())
        if kept_total == 0:
            raise ValueError("no kept candidates — cannot express exec_ok_correct_share")
        return tiers[PreferenceTier.EXEC_OK_CORRECT.value] / kept_total


def build_preference_dataset(
    predictions_by_question: Mapping[str, Sequence[PredictionRecord]],
) -> PreferenceDatasetReport:
    if not predictions_by_question:
        raise ValueError("predictions_by_question is empty — nothing to build a dataset from")
    per_question = tuple(
        build_preference_pairs_for_question(qid, preds)
        for qid, preds in predictions_by_question.items()
    )
    return PreferenceDatasetReport(per_question=per_question)


_HIGH_CORRECT_SHARE_THRESHOLD = 0.7


def assess_expected_dpo_benefit(report: PreferenceDatasetReport) -> str:
    """PLAN.md ★: "这个观察本身写进决策记录，比硬跑一遍有价值" — this
    function computes the real number the observation should be grounded
    in; the observation itself belongs in docs/design-decisions.md, not
    only in a log line."""
    share = report.exec_ok_correct_share
    if share >= _HIGH_CORRECT_SHARE_THRESHOLD:
        return (
            f"exec_ok_correct_share={share:.1%} is already >= "
            f"{_HIGH_CORRECT_SHARE_THRESHOLD:.0%} — PLAN.md: DPO's expected benefit is small "
            f"when the base SFT model already gets most samples right on the first try; "
            f"record this as a decision, not just a training-time observation"
        )
    return (
        f"exec_ok_correct_share={share:.1%} — meaningful room remains for DPO to promote "
        f"tier-2/3 candidates toward tier-1"
    )


def sample_k_candidates(
    sample: NormalizedSample,
    *,
    model_client: ModelClient,
    prompt: str,
    db_root: Path,
    gold_cache: GoldExecCache,
    k: int,
    temperature: float,
    max_tokens: int,
    generate_timeout_s: float,
) -> list[PredictionRecord]:
    """PLAN.md step 1: "对 2000 个问题各采样 k=4 个 SQL". Each of the k
    generations is executed and compared for real via the unchanged A4
    `run_one_sample` — no new execution/comparison logic for B4, only
    the repetition and the downstream tiering."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return [
        run_one_sample(
            sample,
            model_client=model_client,
            prompt=prompt,
            db_root=db_root,
            gold_cache=gold_cache,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_timeout_s=generate_timeout_s,
        )
        for _ in range(k)
    ]


class DpoSamplingConfig(ModelHubBaseConfig):
    """Hashed into `config_hash` on the real run manifest (CLAUDE.md
    §3.1) — PLAN.md item 7 ("采样温度和 k 记进 manifest") needs no
    separate manifest field beyond the normal `config_resolved` capture
    every config already gets."""

    k_samples_per_question: int
    sampling_temperature: float
    max_tokens: int
    generate_timeout_s: float


class DpoTrainingConfig(ModelHubBaseConfig):
    base_sft_adapter_path: str
    lora_target_modules: list[str]
    beta: float
    lr: float
    batch_size: int
    seed: int
    max_hours: float
    output_dir: str


def check_trl_available() -> CheckStatus:
    return CheckStatus.PASS if _TrlDpoTrainer is not None else CheckStatus.SKIP


def build_dpo_config_kwargs(config: DpoTrainingConfig) -> dict[str, object]:
    """Real TRL `DPOConfig` field names — pure and fully testable
    without `trl` installed, unlike the trainer construction below."""
    return {
        "output_dir": config.output_dir,
        "beta": config.beta,
        "learning_rate": config.lr,
        "per_device_train_batch_size": config.batch_size,
        "seed": config.seed,
    }


def run_dpo_training(
    preference_pairs: Sequence[PreferencePair], config: DpoTrainingConfig
) -> NoReturn:
    """Raises (not SKIP-and-continue) if `trl` is not installed — a
    caller must check `check_trl_available()` first (CLAUDE.md §1.3).
    Real construction of the `datasets.Dataset`/model/tokenizer objects
    `DPOTrainer` needs is out of this sandbox's reach entirely (no GPU,
    no `trl`/`transformers`/`peft` install, no B1 adapter on disk) — the
    only branch this function exercises here is the guarded raise."""
    if not preference_pairs:
        raise ValueError("preference_pairs is empty — nothing to train on")
    if not is_available(check_trl_available()):
        raise RuntimeError(
            "cannot start DPO training: trl is not installed — install the `train` extra "
            "on a real GPU machine first"
        )
    from trl import DPOConfig, DPOTrainer  # noqa: F401 — real import, guarded above

    raise NotImplementedError(
        "DPOTrainer construction needs a real loaded base model + B1 LoRA adapter + "
        "tokenizer + a real datasets.Dataset built from preference_pairs — none of which "
        "exist in this sandbox (no GPU, no B1 checkpoint on disk); wire this up against "
        "the real trl API on first real GPU run rather than guessing its exact call shape "
        "untested (CLAUDE.md: 不确定就留 NotImplementedError)"
    )


__all__ = [
    "ClassifiedCandidate",
    "DiscardReason",
    "DpoSamplingConfig",
    "DpoTrainingConfig",
    "PreferenceDatasetReport",
    "PreferencePair",
    "PreferenceTier",
    "QuestionPreferenceResult",
    "assess_expected_dpo_benefit",
    "build_dpo_config_kwargs",
    "build_preference_dataset",
    "build_preference_pairs_for_question",
    "check_trl_available",
    "classify_prediction",
    "run_dpo_training",
    "sample_k_candidates",
]
