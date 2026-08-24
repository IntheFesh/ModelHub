"""Unit tests for eval/model_client.py.

`HttpModelClient` itself needs a served model to hit — real network-bound
behavior, marked `requires_network` and skipped in this sandbox (see
docs/design-decisions.md). What's tested here without a network is
everything that doesn't need one: `GenerationResult` validation and that
`FakeModelClient` (tests/unit/a4/fakes.py) really satisfies the
`ModelClient` Protocol contract.
"""

from __future__ import annotations

import httpx
import pytest
from tests.unit.a4.fakes import fixed_response_client, scripted_client

from modelhub.eval.model_client import GenerationResult, HttpModelClient, ModelClient


def test_generation_result_rejects_unknown_finish_reason() -> None:
    with pytest.raises(ValueError, match="finish_reason"):
        GenerationResult(text="SELECT 1", finish_reason="unknown", model_id="m")  # type: ignore[arg-type]


def test_generation_result_forbids_unknown_field() -> None:
    with pytest.raises(ValueError, match="extra"):
        GenerationResult(
            text="SELECT 1",
            finish_reason="stop",
            model_id="m",
            bogus_field="nope",  # type: ignore[call-arg]
        )


def test_fake_client_satisfies_model_client_protocol() -> None:
    client: ModelClient = fixed_response_client("SELECT 1", finish_reason="stop")
    result = client.generate("prompt", temperature=0.0, max_tokens=64, timeout_s=5.0)
    assert result.text == "SELECT 1"
    assert result.finish_reason == "stop"


def test_scripted_client_keys_by_exact_prompt() -> None:
    client = scripted_client(
        {"s1": GenerationResult(text="SELECT 1", finish_reason="stop", model_id="m")}
    )
    result = client.generate("s1", temperature=0.0, max_tokens=64, timeout_s=5.0)
    assert result.text == "SELECT 1"
    with pytest.raises(KeyError):
        client.generate("s2", temperature=0.0, max_tokens=64, timeout_s=5.0)


def test_fake_client_records_every_prompt_in_order() -> None:
    client = fixed_response_client("SELECT 1")
    client.generate("a", temperature=0.0, max_tokens=1, timeout_s=1.0)
    client.generate("b", temperature=0.0, max_tokens=1, timeout_s=1.0)
    assert client.prompts == ["a", "b"]


@pytest.mark.requires_network
def test_http_model_client_generate_real_call() -> None:
    """Real, non-mocked HTTP round-trip against a served OpenAI/vLLM-
    compatible endpoint. There is no such server in this sandbox (no GPU),
    so this is skipped here and must run on real hardware later."""
    client = HttpModelClient("http://127.0.0.1:8000", model_id="does-not-matter")
    try:
        client.generate("SELECT", temperature=0.0, max_tokens=8, timeout_s=5.0)
    except httpx.ConnectError:
        pytest.skip("no local model server running")
