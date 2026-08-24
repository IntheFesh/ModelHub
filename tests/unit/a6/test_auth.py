"""Unit tests for gateway/auth.py."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode, GatewayError
from modelhub.gateway.auth import AuthConfig, Principal, authenticate, hash_api_key


def _config() -> AuthConfig:
    return AuthConfig.model_validate(
        {
            "principals_by_key_hash": {
                hash_api_key("sk-alice-real-key"): {"tenant_id": "alice", "tier": "standard"},
                hash_api_key("sk-bob-real-key"): {"tenant_id": "bob", "tier": "free"},
            }
        }
    )


def test_authenticate_accepts_a_valid_key() -> None:
    principal = authenticate("sk-alice-real-key", _config())
    assert principal == Principal(tenant_id="alice", tier="standard")


def test_authenticate_rejects_an_unknown_key() -> None:
    with pytest.raises(GatewayError) as exc_info:
        authenticate("sk-not-a-real-key", _config())
    assert exc_info.value.code is ErrorCode.AUTH_INVALID_API_KEY
    assert exc_info.value.retryable is False


def test_authenticate_rejects_empty_key() -> None:
    with pytest.raises(GatewayError):
        authenticate("", _config())


def test_hash_api_key_is_deterministic_and_not_reversible_looking() -> None:
    h1 = hash_api_key("sk-alice-real-key")
    h2 = hash_api_key("sk-alice-real-key")
    assert h1 == h2
    assert h1 != "sk-alice-real-key"
    assert len(h1) == 64  # sha256 hex digest


def test_authenticate_does_not_leak_the_raw_key_into_error_context() -> None:
    with pytest.raises(GatewayError) as exc_info:
        authenticate("sk-super-secret-value", _config())
    assert "sk-super-secret-value" not in str(exc_info.value.context)
    assert "sk-super-secret-value" not in str(exc_info.value)
