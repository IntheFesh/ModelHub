"""Prefix-cache hit rate — the number DD-0001's vLLM-over-SGLang choice
was partly staked on (fixed-schema-prefix prompts should hit vLLM's
prefix cache often; this module is what actually measures that claim
rather than leaving it as an assumption in a design-decision writeup).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixCacheStats:
    hits: int
    total_requests: int

    def __post_init__(self) -> None:
        if self.hits < 0 or self.total_requests < 0:
            raise ValueError(
                f"hits and total_requests must be non-negative, got "
                f"hits={self.hits} total_requests={self.total_requests}"
            )
        if self.hits > self.total_requests:
            raise ValueError(
                f"hits ({self.hits}) cannot exceed total_requests ({self.total_requests})"
            )

    @property
    def hit_rate(self) -> float | None:
        """`None` when there have been no requests yet — never a
        fabricated 0.0 standing in for "no data" (CLAUDE.md §1.1)."""
        if self.total_requests == 0:
            return None
        return self.hits / self.total_requests


__all__ = ["PrefixCacheStats"]
