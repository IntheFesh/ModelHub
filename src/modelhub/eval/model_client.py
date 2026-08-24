"""The model-serving interface the eval runner generates SQL through.

`ModelClient` is a `Protocol`, not an ABC with a mock default: production
wires in `HttpModelClient` (a real OpenAI-compatible HTTP call — untested
in this sandbox, no GPU to serve against, see docs/design-decisions.md);
tests wire in their own fake client living in tests/ (CLAUDE.md §1.4:
mocks belong only in tests/, never imported by src/). This module defines
the contract both sides implement, nothing else.

`finish_reason == "length"` (the OpenAI/vLLM convention for "hit
max_tokens") is how the runner detects CLAUDE.md §2.4's OUTPUT_TRUNCATED
case — decided here, before the (possibly incomplete) SQL text is ever
handed to sqlexec.
"""

from __future__ import annotations

from typing import Literal, Protocol

from modelhub.common.config import ModelHubBaseConfig


class GenerationResult(ModelHubBaseConfig):
    text: str
    finish_reason: Literal["stop", "length"]
    model_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelClient(Protocol):
    def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, timeout_s: float
    ) -> GenerationResult: ...


class HttpModelClient:
    """OpenAI/vLLM-compatible `/v1/completions` client.

    Untested in this sandbox (no GPU, no served model to hit) — see
    docs/design-decisions.md DD-0003. Real code, marked requires_gpu at
    the test level, not replaced by a fake here.
    """

    def __init__(self, base_url: str, *, model_id: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key

    def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, timeout_s: float
    ) -> GenerationResult:
        import httpx

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = httpx.post(
            f"{self._base_url}/v1/completions",
            json={
                "model": self._model_id,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers=headers,
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        return GenerationResult(
            text=choice["text"],
            finish_reason=choice["finish_reason"],
            model_id=payload.get("model", self._model_id),
            prompt_tokens=payload.get("usage", {}).get("prompt_tokens"),
            completion_tokens=payload.get("usage", {}).get("completion_tokens"),
        )
