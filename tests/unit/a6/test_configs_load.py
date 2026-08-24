"""Every configs/gateway/*.yaml file must actually load against its
schema — a config that only ever gets constructed inline in tests would
leave the real committed YAML unverified."""

from __future__ import annotations

from pathlib import Path

from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.gateway.circuit_breaker import CircuitBreakerConfig
from modelhub.gateway.quota import QuotaConfig
from modelhub.gateway.rate_limit import RateLimitConfig

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "gateway"


def test_rate_limit_config_loads() -> None:
    config, config_hash = load_yaml_config(
        RateLimitConfig, _CONFIGS_DIR / "rate_limit.yaml", stage=Stage.SERVE
    )
    assert config.requests_per_window > 0
    assert config.window_seconds > 0
    assert config_hash.startswith("sha256:")


def test_quota_config_loads() -> None:
    config, _ = load_yaml_config(QuotaConfig, _CONFIGS_DIR / "quota.yaml", stage=Stage.SERVE)
    assert "free" in config.daily_token_quota_by_tier
    assert config.daily_token_quota_by_tier["free"] > 0


def test_circuit_breaker_config_loads() -> None:
    config, _ = load_yaml_config(
        CircuitBreakerConfig, _CONFIGS_DIR / "circuit_breaker.yaml", stage=Stage.SERVE
    )
    assert config.failure_threshold > 0
    assert config.cooldown_seconds > 0
