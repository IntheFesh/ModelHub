"""Unit tests for gateway/routing.py."""

from __future__ import annotations

from pathlib import Path

from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.gateway.routing import RoutingConfig, estimate_token_count, route_by_schema_length

_ROUTING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "gateway" / "routing.yaml"


def _config(threshold: int = 100) -> RoutingConfig:
    return RoutingConfig(
        schema_token_threshold=threshold,
        threshold_source="estimate",
        short_schema_model_id="arctic-text2sql-r1-7b",
        long_schema_model_id="qwen3.5-9b",
    )


def test_estimate_token_count_is_roughly_chars_over_four() -> None:
    assert estimate_token_count("x" * 400) == 100
    assert estimate_token_count("") == 0


def test_short_schema_routes_to_short_schema_model() -> None:
    model_id = route_by_schema_length("x" * 40, _config(threshold=100))
    assert model_id == "arctic-text2sql-r1-7b"


def test_long_schema_routes_to_long_schema_model() -> None:
    model_id = route_by_schema_length("x" * 800, _config(threshold=100))
    assert model_id == "qwen3.5-9b"


def test_routing_is_a_hard_threshold_not_a_probability() -> None:
    config = _config(threshold=100)
    just_under = route_by_schema_length("x" * 399, config)  # 99 tokens
    just_over = route_by_schema_length("x" * 400, config)  # 100 tokens
    assert just_under == "arctic-text2sql-r1-7b"
    assert just_over == "qwen3.5-9b"


def test_real_routing_config_loads_and_is_labeled_an_estimate() -> None:
    config, _ = load_yaml_config(RoutingConfig, _ROUTING_CONFIG_PATH, stage=Stage.SERVE)
    assert config.threshold_source == "estimate"
    assert config.short_schema_model_id == "arctic-text2sql-r1-7b"
    assert config.long_schema_model_id == "qwen3.5-9b"
