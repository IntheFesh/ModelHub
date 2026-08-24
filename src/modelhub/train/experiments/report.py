"""B2's selection table + effect-cost narrative sentence.

★ PLAN.md's two hard framing rules, both enforced as code here rather
than left to whoever writes the report to remember:

1. "这是技术选型对比不是消融实验，表述别用「消融显示」" —
   `format_effect_cost_sentence` never emits that phrase; it produces
   PLAN.md's own template shape ("X 相比 Y：... 两周迭代一次的节奏选
   X。").
2. "准确率只有走完收敛的那一个模型才有意义，短跑的准确率不要拿来比" —
   `ComparisonEntry.provenance` is a required field (`MetricProvenance`,
   not an optional flag), and `format_effect_cost_sentence` REQUIRES an
   explicit `accuracy_delta_points: float | None` rather than silently
   omitting the accuracy clause — passing `None` (this project's honest
   state for every group in this GPU-less sandbox, since nothing here
   ran to convergence) produces a sentence that says so explicitly
   instead of a sentence that quietly never mentions accuracy at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelhub.train.experiments.common import TrainingRunMetrics


class MetricProvenance(StrEnum):
    SHORT_RUN = "SHORT_RUN"  # 200-500 step comparison run — memory/throughput/cost only
    CONVERGENCE_RUN = "CONVERGENCE_RUN"  # trained to convergence — accuracy is meaningful here


@dataclass(frozen=True)
class ComparisonEntry:
    variant_name: str
    metrics: TrainingRunMetrics
    num_steps: int
    provenance: MetricProvenance

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

    @property
    def total_time_s(self) -> float:
        return self.metrics.step_time_s * self.num_steps


def render_selection_table(entries: Sequence[ComparisonEntry]) -> str:
    if not entries:
        raise ValueError("cannot render a selection table with zero entries")
    header = (
        "| variant | peak_memory_gb | throughput_tok_s | step_time_s | "
        "trainable_params | total_cost | provenance |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for entry in entries:
        m = entry.metrics
        rows.append(
            f"| {entry.variant_name} | {m.peak_memory_bytes / 1e9:.2f} | "
            f"{m.throughput_tokens_per_s:.1f} | {m.step_time_s:.3f} | "
            f"{m.trainable_param_count:,} | ¥{m.total_cost:.2f} | {entry.provenance.value} |"
        )
    return "\n".join(rows)


def format_effect_cost_sentence(
    *,
    baseline: ComparisonEntry,
    candidate: ComparisonEntry,
    accuracy_delta_points: float | None,
    conclusion: str,
) -> str:
    """`accuracy_delta_points` is `candidate - baseline`, in execution-
    accuracy percentage points, from a real convergence run — pass
    `None` (this project's honest state everywhere in this sandbox) to
    get an explicit "尚无收敛跑数据" clause instead of a silently
    omitted one."""
    memory_pct = _pct_delta(baseline.metrics.peak_memory_bytes, candidate.metrics.peak_memory_bytes)
    memory_clause = f"显存峰值{'省' if memory_pct < 0 else '增'} {abs(memory_pct):.1f}%"

    if accuracy_delta_points is None:
        accuracy_clause = "EX 差多少个点尚无收敛跑数据(PLAN.md: 短跑准确率不可比)"
    else:
        sign = "+" if accuracy_delta_points >= 0 else ""
        accuracy_clause = f"EX 差 {sign}{accuracy_delta_points:.1f} 个点"

    # ASCII punctuation deliberately used throughout (not the fullwidth
    # ，。： PLAN.md's own example sentence uses) — ruff's RUF001 flags
    # ambiguous fullwidth punctuation project-wide and this project's
    # established fix (train/smoke_test.py, this same round) is to
    # rephrase around it rather than add a per-line suppression.
    return (
        f"{candidate.variant_name} 相比 {baseline.variant_name}: {accuracy_clause}, "
        f"{memory_clause}, 训练时间从 {baseline.total_time_s:.0f}s 降到 "
        f"{candidate.total_time_s:.0f}s, 单次成本从 ¥{baseline.metrics.total_cost:.2f} "
        f"降到 ¥{candidate.metrics.total_cost:.2f}. {conclusion}"
    )


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("cannot express a percentage delta against a zero baseline")
    return (candidate - baseline) / baseline * 100


__all__ = [
    "ComparisonEntry",
    "MetricProvenance",
    "format_effect_cost_sentence",
    "render_selection_table",
]
