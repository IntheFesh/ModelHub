"""Schema-token-length routing between the platform's two serving models.

MODEL-SELECTION-FINAL.md §五 (A6's amendment): the routing判据 is not
"simple vs. complex" — it's a schema-token-length threshold, because
FACTS.md §二's KV-economics table shows the two models' per-prompt-length
concurrency curves cross somewhere between 1k and 3k tokens (Arctic-7B's
plain-GQA KV grows linearly with context; Qwen3.5-9B's GDN-hybrid KV
barely grows at all, so it wins once the schema is long enough).

★ `schema_token_threshold` in `configs/gateway/routing.yaml` is currently
an ESTIMATE (`threshold_source: "estimate"`) seeded from that 1k–3k range,
not yet the real measured crossing point — A8's prompt-length bench sweep
is what determines the actual value (PLAN.md: "路由阈值回填
configs/gateway/"). Routing code must never assume its own config is the
final number; `RoutingConfig.threshold_source` exists specifically so a
report or gate can tell a bench-measured threshold from a placeholder one
at a glance, and refuse to treat an estimate-sourced routing decision as
production-grade if that ever matters (A9's gate).
"""

from __future__ import annotations

from typing import Literal

from modelhub.common.config import ModelHubBaseConfig

# Rule-of-thumb English-text token estimate (~4 chars/token) — the same
# kind of documented, clearly-labeled approximation as
# serve/schema_format.py's other estimates. A real tokenizer call (the
# actual model's tokenizer) is the correct replacement once one is wired
# up to a served model; using it here would require a live model/tokenizer
# process this sandbox does not have.
_CHARS_PER_TOKEN_ESTIMATE = 4


class RoutingConfig(ModelHubBaseConfig):
    schema_token_threshold: int
    threshold_source: Literal["estimate", "measured"]
    short_schema_model_id: str
    long_schema_model_id: str


def estimate_token_count(text: str) -> int:
    """Approximate token count via `len(text) // 4`. NOT a real
    tokenizer — see module docstring. Never off by a factor large enough
    to flip a routing decision that isn't already near the threshold, but
    genuinely wrong in the small (a few dozen token) sense a real BPE
    tokenizer would confirm or correct."""
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


def route_by_schema_length(schema_text: str, config: RoutingConfig) -> str:
    """Return the `model_id` this request should be served by."""
    token_count = estimate_token_count(schema_text)
    if token_count < config.schema_token_threshold:
        return config.short_schema_model_id
    return config.long_schema_model_id


__all__ = ["RoutingConfig", "estimate_token_count", "route_by_schema_length"]
