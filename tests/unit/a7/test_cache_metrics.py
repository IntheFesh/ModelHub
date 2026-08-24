"""Unit tests for monitor/cache_metrics.py."""

from __future__ import annotations

import pytest

from modelhub.monitor.cache_metrics import PrefixCacheStats


def test_hit_rate_basic() -> None:
    stats = PrefixCacheStats(hits=30, total_requests=100)
    assert stats.hit_rate == pytest.approx(0.3)


def test_hit_rate_is_none_when_no_requests() -> None:
    stats = PrefixCacheStats(hits=0, total_requests=0)
    assert stats.hit_rate is None


def test_all_hits_is_rate_one() -> None:
    stats = PrefixCacheStats(hits=10, total_requests=10)
    assert stats.hit_rate == 1.0


def test_negative_hits_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PrefixCacheStats(hits=-1, total_requests=10)


def test_negative_total_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PrefixCacheStats(hits=0, total_requests=-1)


def test_hits_exceeding_total_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        PrefixCacheStats(hits=11, total_requests=10)
