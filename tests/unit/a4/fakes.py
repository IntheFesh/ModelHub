"""Fake `ModelClient` and manifest builder for eval/ tests.

CLAUDE.md §1.4: mocks live only in `tests/` and are never imported by
`src/`. `FakeModelClient` is a real implementation of the `ModelClient`
Protocol (not a monkeypatch or a stub that fakes sqlexec/compare
behavior) — it exists because there is no served model to hit in this
sandbox (no GPU). Every other stage a test using this fake exercises (SQL
execution, comparison) runs for real.

`make_valid_manifest` builds a `RunManifest` with every A4-required field
(`eval/report.py`'s `REQUIRED_MANIFEST_FIELDS`) filled with a plausible
value, so unit and meta tests for `report.py` only need to override the
one field they're actually testing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.eval.model_client import GenerationResult


class FakeModelClient:
    """Records every prompt it was asked to generate for, in order."""

    def __init__(self, respond: Callable[[str], GenerationResult]) -> None:
        self._respond = respond
        self.prompts: list[str] = []

    def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, timeout_s: float
    ) -> GenerationResult:
        self.prompts.append(prompt)
        return self._respond(prompt)


def scripted_client(responses: dict[str, GenerationResult]) -> FakeModelClient:
    """A client keyed by exact prompt text — tests pass `prompt_builder=lambda
    s: s.sample_id` so the key is just the sample_id, no prompt-template
    concerns (that's A5's job, not eval/'s)."""

    def respond(prompt: str) -> GenerationResult:
        if prompt not in responses:
            raise KeyError(f"no scripted response for prompt {prompt!r}")
        return responses[prompt]

    return FakeModelClient(respond)


def fixed_response_client(
    text: str,
    *,
    finish_reason: Literal["stop", "length"] = "stop",
    model_id: str = "fake-model",
) -> FakeModelClient:
    result = GenerationResult(text=text, finish_reason=finish_reason, model_id=model_id)
    return FakeModelClient(lambda _prompt: result)


def make_valid_manifest(**overrides: object) -> RunManifest:
    payload: dict[str, object] = {
        "run_id": "20260824-0000-testrun",
        "git_sha": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "git_dirty": False,
        "dataset_version": "bird23-filtered@v1",
        "dataset_hash": "sha256:0" * 8,
        "eval_tier": "quick",
        "model_id": "fake-model",
        "comparator_version": "2.1.0",
        "seed": 42,
        "n_samples": 3,
        "contaminated": False,
        "status": RunStatus.COMPLETED,
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)
