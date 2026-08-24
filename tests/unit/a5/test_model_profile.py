"""Unit tests for serve/model_profile.py.

`kv_bytes_per_token` values are checked against FACTS.md §二's documented
KV/token numbers exactly (56 KiB Arctic-7B, 32 KiB Qwen3.5-9B) — those are
architecture facts (layer/kv_head/head_dim counts), independently
reproducible from the formula, not values this test just copies from the
implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.common.errors import ConfigError
from modelhub.serve.model_profile import (
    ModelProfile,
    gpu_blocks_to_cacheable_tokens,
    kv_bytes_per_token,
    kv_kib_per_token,
    load_model_profile,
)

_PROFILES_DIR = Path(__file__).resolve().parents[3] / "configs" / "serve" / "model_profiles"


def test_arctic_7b_profile_matches_facts_md_kv_per_token() -> None:
    profile, _ = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    assert kv_bytes_per_token(profile) == 56 * 1024
    assert kv_kib_per_token(profile) == pytest.approx(56.0)


def test_qwen3_5_9b_profile_matches_facts_md_kv_per_token() -> None:
    profile, _ = load_model_profile(_PROFILES_DIR / "qwen3_5_9b.yaml")
    assert kv_bytes_per_token(profile) == 32 * 1024
    assert kv_kib_per_token(profile) == pytest.approx(32.0)


def test_qwen3_5_9b_has_gdn_layers_arctic_does_not() -> None:
    arctic, _ = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    qwen, _ = load_model_profile(_PROFILES_DIR / "qwen3_5_9b.yaml")
    assert arctic.gdn_layers == 0
    assert arctic.gdn_fixed_state_bytes == 0
    assert qwen.gdn_layers > 0
    assert qwen.gdn_fixed_state_bytes > 0


def test_load_model_profile_returns_deterministic_hash() -> None:
    _, hash1 = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    _, hash2 = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    assert hash1 == hash2
    assert hash1.startswith("sha256:")


def test_different_profiles_hash_differently() -> None:
    _, arctic_hash = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    _, qwen_hash = load_model_profile(_PROFILES_DIR / "qwen3_5_9b.yaml")
    assert arctic_hash != qwen_hash


def test_model_profile_forbids_unknown_field() -> None:
    with pytest.raises(ValueError, match="extra"):
        ModelProfile.model_validate(
            {
                "model_id": "x",
                "total_layers": 1,
                "full_attention_layers": 1,
                "gdn_layers": 0,
                "kv_heads": 1,
                "head_dim": 1,
                "kv_dtype_bytes": 2,
                "gdn_fixed_state_bytes": 0,
                "weight_bytes": 1,
                "bogus_field": "nope",
            }
        )


def test_load_model_profile_rejects_missing_field(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("model_id: incomplete\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_model_profile(bad_config)


def test_gpu_blocks_to_cacheable_tokens() -> None:
    assert gpu_blocks_to_cacheable_tokens(100, 16) == 1600
    assert gpu_blocks_to_cacheable_tokens(0, 16) == 0
