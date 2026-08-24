"""Unit tests for gateway/billing.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.gateway.billing import BillingConfig, compute_request_cost


def _config() -> BillingConfig:
    return BillingConfig(
        price_per_1k_prompt_tokens_usd={"model-a": 0.001},
        price_per_1k_completion_tokens_usd={"model-a": 0.002},
    )


def test_compute_request_cost_arithmetic() -> None:
    cost = compute_request_cost(
        "model-a", prompt_tokens=2000, completion_tokens=500, config=_config()
    )
    assert cost.prompt_cost_usd == pytest.approx(0.002)
    assert cost.completion_cost_usd == pytest.approx(0.001)
    assert cost.total_cost_usd == pytest.approx(0.003)


def test_zero_tokens_costs_nothing() -> None:
    cost = compute_request_cost("model-a", prompt_tokens=0, completion_tokens=0, config=_config())
    assert cost.total_cost_usd == 0.0


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_request_cost("model-a", prompt_tokens=-1, completion_tokens=0, config=_config())


def test_unknown_model_id_rejected() -> None:
    with pytest.raises(ValueError, match="model_id"):
        compute_request_cost(
            "no-such-model", prompt_tokens=1, completion_tokens=1, config=_config()
        )


def test_real_billing_config_loads_and_covers_both_serving_models() -> None:
    path = Path(__file__).resolve().parents[3] / "configs" / "gateway" / "billing.yaml"
    config, _ = load_yaml_config(BillingConfig, path, stage=Stage.SERVE)
    for model_id in ("arctic-text2sql-r1-7b", "qwen3.5-9b"):
        cost = compute_request_cost(
            model_id, prompt_tokens=1000, completion_tokens=1000, config=config
        )
        assert cost.total_cost_usd > 0
