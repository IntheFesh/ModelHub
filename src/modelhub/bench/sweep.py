"""2D load-test sweep: concurrency x prompt-length, per model — the shape
PLAN.md calls for ("二维压测 sweep（并发 × prompt 长度 × 2 模型）").
Orchestrates `gpu_guard` -> repeated `load_test.run_load_test` calls ->
a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.bench.gpu_guard import GpuExclusivityCheck, guard_gpu_exclusivity
from modelhub.bench.load_test import LoadTestConfig, LoadTestResult, run_load_test
from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.model_client import ModelClient


class SweepGridConfig(ModelHubBaseConfig):
    concurrency_levels: list[int]
    prompt_length_labels: list[str]
    num_requests_per_cell: int
    max_tokens: int
    temperature: float
    generate_timeout_s: float


@dataclass(frozen=True)
class SweepCell:
    concurrency: int
    prompt_length_label: str
    prompt: str


@dataclass(frozen=True)
class SweepReport:
    model_id: str
    results: tuple[LoadTestResult, ...]
    # None means exclusivity was never checked for this report (only
    # legitimate when a test explicitly bypassed the guard) — never a
    # fabricated True/False standing in for "we didn't actually check."
    gpu_exclusivity_check: GpuExclusivityCheck | None

    def result_for(self, *, concurrency: int, prompt_length_label: str) -> LoadTestResult | None:
        for r in self.results:
            if r.concurrency == concurrency and r.prompt_length_label == prompt_length_label:
                return r
        return None


def run_sweep(
    model_client: ModelClient,
    *,
    model_id: str,
    cells: list[SweepCell],
    num_requests_per_cell: int,
    max_tokens: int,
    temperature: float,
    generate_timeout_s: float,
    gpu_index: int = 0,
    skip_gpu_guard: bool = False,
) -> SweepReport:
    """Run `run_load_test` once per cell in `cells`.

    Refuses to start unless GPU exclusivity is confirmed (CLAUDE.md
    Section 5.2) — `skip_gpu_guard` exists only so tests can exercise the
    sweep orchestration itself against a fake client with no real GPU
    contention to check; it must never be set True for a real bench run
    (`guard_gpu_exclusivity` already refuses to run on anything short of
    a confirmed-clean PASS, including "couldn't check").
    """
    exclusivity_check: GpuExclusivityCheck | None = None
    if not skip_gpu_guard:
        exclusivity_check = guard_gpu_exclusivity(gpu_index)

    results = []
    for cell in cells:
        config = LoadTestConfig(
            concurrency=cell.concurrency,
            num_requests=num_requests_per_cell,
            max_tokens=max_tokens,
            temperature=temperature,
            generate_timeout_s=generate_timeout_s,
            prompt_length_label=cell.prompt_length_label,
        )
        results.append(run_load_test(model_client, cell.prompt, config))

    return SweepReport(
        model_id=model_id, results=tuple(results), gpu_exclusivity_check=exclusivity_check
    )


__all__ = ["SweepCell", "SweepGridConfig", "SweepReport", "run_sweep"]
