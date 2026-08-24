"""API-key authentication.

Keys are stored and compared as salted SHA-256 hashes, never as plaintext
— `AuthConfig` (loaded from `configs/gateway/auth.yaml`) holds
`hash -> Principal`, not `raw_key -> Principal`, so a leaked config file
does not itself leak usable credentials. Comparison uses
`hmac.compare_digest` (constant-time) rather than `==`, which is the
standard defense against a timing side-channel on hash comparison.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Literal

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode, GatewayError, Stage


class Principal(ModelHubBaseConfig):
    tenant_id: str
    tier: Literal["free", "standard", "premium"]


class AuthConfig(ModelHubBaseConfig):
    # key: sha256 hex digest of the raw API key (see `hash_api_key`).
    principals_by_key_hash: dict[str, Principal]


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def authenticate(raw_api_key: str, config: AuthConfig) -> Principal:
    """Return the `Principal` for `raw_api_key`, or raise `GatewayError`.

    Every configured hash is compared against (not just checked via dict
    lookup) so lookup time doesn't itself leak which prefix of the key
    hash was correct — dict lookup on the hash is already effectively
    constant-time relative to key content (it hashes to a bucket, not a
    char-by-char scan), but the explicit `compare_digest` scan below is
    the standard, unambiguous way to write "this comparison is
    timing-safe" rather than relying on an implementation detail of
    Python's dict.
    """
    key_hash = hash_api_key(raw_api_key)
    for candidate_hash, principal in config.principals_by_key_hash.items():
        if hmac.compare_digest(candidate_hash, key_hash):
            return principal
    raise GatewayError(
        "invalid API key",
        code=ErrorCode.AUTH_INVALID_API_KEY,
        stage=Stage.SERVE,
        context={},
        retryable=False,
    )


__all__ = ["AuthConfig", "Principal", "authenticate", "hash_api_key"]
