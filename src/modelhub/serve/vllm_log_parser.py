"""Parse real vLLM startup log text into structured facts for
`RunManifest` population.

MODEL-SELECTION-FINAL.md §五 (A5's amendment): "启动后从 vLLM 日志分别
解析两者的 KV block 数与 GDN 状态占用写进 manifest" — the measured KV
block count (not the estimated `kv_bytes_per_token` formula in
`model_profile.py`) is what should actually end up in a served run's
manifest, because the estimate can be wrong (weight size, `gpu_memory_
utilization`, and runtime overhead are all only approximately known ahead
of time) while the log is vLLM reporting what it actually allocated.

★ Honesty note (CLAUDE.md §12): the `GPU blocks` / `Maximum concurrency` /
`Loading model weights took` patterns below match vLLM's long-stable,
well-known log line formats (present across many released versions) and
are ported here with reasonable confidence. The GDN-state-footprint
pattern is NOT similarly confirmed — Qwen3.5/GDN support in vLLM is new
enough that no specific, stable log line format for it could be verified
in this sandbox (no network access to vLLM's GDN-support source, no GPU
to run it and observe real output). That pattern is a best-effort,
clearly-flagged guess, and `gdn_state_footprint_bytes` MUST be treated as
unconfirmed until checked against a real run's actual log output — this
is exactly the kind of gap CLAUDE.md §12 says to name outright rather
than paper over with false confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_GPU_BLOCKS_RE = re.compile(r"#\s*GPU blocks:\s*(\d+),\s*#\s*CPU blocks:\s*(\d+)")
_MAX_CONCURRENCY_RE = re.compile(r"Maximum concurrency for (\d+) tokens per request:\s*([\d.]+)x")
_WEIGHT_LOAD_RE = re.compile(r"Loading model weights took\s*([\d.]+)\s*GB")
# UNCONFIRMED pattern — see module docstring.
_GDN_STATE_RE = re.compile(
    r"(?:GDN|gated[- _]?delta|recurrent)[^\n]*state[^\n]*?(\d+(?:\.\d+)?)\s*(GiB|MiB|GB|MB|bytes)",
    re.IGNORECASE,
)

_UNIT_TO_BYTES = {
    "bytes": 1,
    "mb": 1_000_000,
    "mib": 1024**2,
    "gb": 1_000_000_000,
    "gib": 1024**3,
}


@dataclass(frozen=True)
class VllmStartupFacts:
    gpu_blocks: int | None = None
    cpu_blocks: int | None = None
    max_concurrency_tokens: int | None = None
    max_concurrency: float | None = None
    weight_load_gb: float | None = None
    # unconfirmed pattern (see module docstring) — None means "not found
    # in this log", which given the pattern's own unreliability is not
    # strong evidence of absence either way. Never treat this as 0.
    gdn_state_footprint_bytes: int | None = None
    matched_lines: tuple[str, ...] = field(default_factory=tuple)


def parse_vllm_startup_log(log_text: str) -> VllmStartupFacts:
    """Extract whatever facts are present in `log_text`. Every field is
    `None` if its pattern wasn't found — never a fabricated 0 or guess
    (CLAUDE.md §1.1: missing is `None`, not 0)."""
    matched: list[str] = []

    gpu_blocks = cpu_blocks = None
    if m := _GPU_BLOCKS_RE.search(log_text):
        gpu_blocks, cpu_blocks = int(m.group(1)), int(m.group(2))
        matched.append(m.group(0))

    max_concurrency_tokens = max_concurrency = None
    if m := _MAX_CONCURRENCY_RE.search(log_text):
        max_concurrency_tokens = int(m.group(1))
        max_concurrency = float(m.group(2))
        matched.append(m.group(0))

    weight_load_gb = None
    if m := _WEIGHT_LOAD_RE.search(log_text):
        weight_load_gb = float(m.group(1))
        matched.append(m.group(0))

    gdn_state_footprint_bytes = None
    if m := _GDN_STATE_RE.search(log_text):
        value = float(m.group(1))
        unit = m.group(2).lower()
        gdn_state_footprint_bytes = int(value * _UNIT_TO_BYTES[unit])
        matched.append(m.group(0))

    return VllmStartupFacts(
        gpu_blocks=gpu_blocks,
        cpu_blocks=cpu_blocks,
        max_concurrency_tokens=max_concurrency_tokens,
        max_concurrency=max_concurrency,
        weight_load_gb=weight_load_gb,
        gdn_state_footprint_bytes=gdn_state_footprint_bytes,
        matched_lines=tuple(matched),
    )


__all__ = ["VllmStartupFacts", "parse_vllm_startup_log"]
