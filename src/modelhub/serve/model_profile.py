"""Per-model KV-cache economics — the arithmetic MODEL-SELECTION-FINAL.md
§二's "三模型 KV 经济学" table is derived from, as real (config-driven,
testable) code instead of a table someone has to keep re-deriving by hand.

    KV/token = 2(K,V) × 全注意力层数 × KV头数 × head_dim × 2 bytes

`full_attention_layers` (not `total_layers`) is the term that matters for
a GDN hybrid model: Qwen3.5-9B's 24 GDN layers keep an O(1) fixed-size
state, not a per-token KV entry, so only its 8 full-attention layers
contribute to `kv_bytes_per_token` — this is exactly the fact that makes
Qwen3.5's KV/token (32 KiB) come out lower than Arctic-7B's (56 KiB)
despite Qwen3.5 having more layers overall.

Concrete profile values live in `configs/serve/model_profiles/*.yaml`
(CLAUDE.md §4: hyperparameters belong in configs/, not magic numbers in
code) — this module only defines the schema and the formula.
"""

from __future__ import annotations

from pathlib import Path

from modelhub.common.config import ModelHubBaseConfig, load_yaml_config
from modelhub.common.errors import Stage

_KIB = 1024


class ModelProfile(ModelHubBaseConfig):
    model_id: str
    total_layers: int
    full_attention_layers: int
    gdn_layers: int
    kv_heads: int
    head_dim: int
    kv_dtype_bytes: int
    # GDN's per-layer fixed-size recurrent state, independent of sequence
    # length — 0 for a non-hybrid (pure full-attention) model. Documented
    # as an *estimate* pending real measurement — see
    # docs/design-decisions.md and MODEL-SELECTION-FINAL.md §六.
    gdn_fixed_state_bytes: int
    weight_bytes: int


def kv_bytes_per_token(profile: ModelProfile) -> int:
    """`2(K,V) × full_attention_layers × kv_heads × head_dim × kv_dtype_bytes`."""
    return (
        2
        * profile.full_attention_layers
        * profile.kv_heads
        * profile.head_dim
        * profile.kv_dtype_bytes
    )


def kv_kib_per_token(profile: ModelProfile) -> float:
    return kv_bytes_per_token(profile) / _KIB


def gpu_blocks_to_cacheable_tokens(num_gpu_blocks: int, block_size_tokens: int) -> int:
    """What a vLLM startup log's `# GPU blocks: N` actually means in
    tokens of KV cache capacity — the unit `vllm_log_parser.py` extracts
    from a real log and this function converts into the quantity
    everything else in this project (capacity planning, manifests) cares
    about."""
    return num_gpu_blocks * block_size_tokens


def load_model_profile(path: Path) -> tuple[ModelProfile, str]:
    """Load a `ModelProfile` from a `configs/serve/model_profiles/*.yaml`
    file. Returns `(profile, config_hash)` — the hash goes into the run
    manifest's `config_hash` for any run that served this model."""
    return load_yaml_config(ModelProfile, path, stage=Stage.SERVE)


__all__ = [
    "ModelProfile",
    "gpu_blocks_to_cacheable_tokens",
    "kv_bytes_per_token",
    "kv_kib_per_token",
    "load_model_profile",
]
