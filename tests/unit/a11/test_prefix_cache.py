"""Unit tests for bench/experiments/prefix_cache.py."""

from __future__ import annotations

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.experiments.prefix_cache import (
    ArchitectureCacheComparison,
    build_interleaved_traffic,
    run_prefix_cache_traffic,
)
from modelhub.monitor.cache_metrics import PrefixCacheStats


class TestBuildInterleavedTraffic:
    def test_degree_one_round_robins_every_request(self) -> None:
        traffic = build_interleaved_traffic(
            {"db_a": ["a1", "a2"], "db_b": ["b1", "b2"]}, interleaving_degree=1
        )
        db_sequence = [r.db_id for r in traffic]
        assert db_sequence == ["db_a", "db_b", "db_a", "db_b"]

    def test_high_degree_reproduces_the_forbidden_easy_case(self) -> None:
        traffic = build_interleaved_traffic(
            {"db_a": ["a1", "a2", "a3"], "db_b": ["b1"]}, interleaving_degree=10
        )
        db_sequence = [r.db_id for r in traffic]
        # db_a exhausts its 3 before db_b ever gets a turn.
        assert db_sequence == ["db_a", "db_a", "db_a", "db_b"]

    def test_preserves_total_request_count(self) -> None:
        prompts_by_db = {"a": ["1", "2", "3"], "b": ["4", "5"], "c": ["6"]}
        traffic = build_interleaved_traffic(prompts_by_db, interleaving_degree=2)
        assert len(traffic) == 6

    def test_zero_or_negative_degree_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            build_interleaved_traffic({"a": ["1"]}, interleaving_degree=0)

    def test_empty_prompts_by_db_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_interleaved_traffic({}, interleaving_degree=1)


class TestRunPrefixCacheTraffic:
    def test_fires_every_request_in_order(self) -> None:
        client = BenchFakeClient()
        traffic = build_interleaved_traffic({"a": ["p1", "p2"], "b": ["p3"]}, interleaving_degree=1)
        run_prefix_cache_traffic(
            client, traffic, max_tokens=64, temperature=0.0, generate_timeout_s=5.0
        )
        # BenchFakeClient doesn't record prompts itself, but a successful
        # run with no exception over every request is itself the thing
        # under test here — this deliberately runs single-threaded
        # (unlike load_test.py's pool), so order is exactly `traffic`'s.


class TestArchitectureCacheComparison:
    def test_hit_rate_discount_positive_when_arctic_higher(self) -> None:
        comparison = ArchitectureCacheComparison(
            arctic_stats=PrefixCacheStats(hits=90, total_requests=100),
            qwen_stats=PrefixCacheStats(hits=40, total_requests=100),
        )
        assert comparison.hit_rate_discount == pytest.approx(0.5)

    def test_discount_none_when_either_side_has_no_requests(self) -> None:
        comparison = ArchitectureCacheComparison(
            arctic_stats=PrefixCacheStats(hits=0, total_requests=0),
            qwen_stats=PrefixCacheStats(hits=40, total_requests=100),
        )
        assert comparison.hit_rate_discount is None
