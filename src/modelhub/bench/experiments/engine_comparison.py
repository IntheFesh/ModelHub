"""Group 5: inference engine comparison — vLLM vs SGLang, same prompt
distribution, same hardware, same SLA.

TensorRT-LLM is deliberately excluded: on this project's SM120/121
target, trtllm-gen's FMHA cubin is missing and the runtime falls back to
an unfused MHA kernel, and installation only works via an NGC container
(not a pip/uv-installable package this project's tooling can pin the same
way as everything else) — see DD-0025.

Both vLLM and SGLang expose an OpenAI-compatible `/v1/completions`
endpoint, so this experiment reuses `eval/model_client.py::HttpModelClient`
(A4) unchanged for both sides — there is no engine-specific client to
write, only two different `base_url`s pointed at `bench/load_test.py::
run_load_test` (A8).
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.bench.load_test import LoadTestConfig, LoadTestResult, run_load_test
from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.model_client import ModelClient


class EngineComparisonExperimentConfig(ModelHubBaseConfig):
    concurrency: int
    num_requests: int
    max_tokens: int
    temperature: float
    generate_timeout_s: float
    prompt_length_label: str
    sla_latency_p99_s: float


@dataclass(frozen=True)
class EngineComparisonResult:
    vllm_result: LoadTestResult
    sglang_result: LoadTestResult
    sla_latency_p99_s: float

    @property
    def vllm_within_sla(self) -> bool | None:
        if self.vllm_result.latency_p99_s is None:
            return None
        return self.vllm_result.latency_p99_s <= self.sla_latency_p99_s

    @property
    def sglang_within_sla(self) -> bool | None:
        if self.sglang_result.latency_p99_s is None:
            return None
        return self.sglang_result.latency_p99_s <= self.sla_latency_p99_s

    @property
    def qps_ratio_sglang_over_vllm(self) -> float | None:
        if self.vllm_result.qps == 0:
            return None
        return self.sglang_result.qps / self.vllm_result.qps


def run_engine_comparison(
    *,
    vllm_client: ModelClient,
    sglang_client: ModelClient,
    prompt: str,
    config: EngineComparisonExperimentConfig,
) -> EngineComparisonResult:
    load_config = LoadTestConfig(
        concurrency=config.concurrency,
        num_requests=config.num_requests,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        generate_timeout_s=config.generate_timeout_s,
        prompt_length_label=config.prompt_length_label,
    )
    return EngineComparisonResult(
        vllm_result=run_load_test(vllm_client, prompt, load_config),
        sglang_result=run_load_test(sglang_client, prompt, load_config),
        sla_latency_p99_s=config.sla_latency_p99_s,
    )


__all__ = ["EngineComparisonExperimentConfig", "EngineComparisonResult", "run_engine_comparison"]
