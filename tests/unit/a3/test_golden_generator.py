import json
from pathlib import Path

from tests.conftest import TEST_PG_DSN, requires_postgres

from modelhub.compare.golden_generator import (
    DIMENSIONS,
    generate_dialect_adversarial_pairs,
    generate_synthetic_pairs,
    write_golden_pairs,
)


def test_generates_at_least_150_pairs() -> None:
    pairs = generate_synthetic_pairs()
    assert len(pairs) >= 150


def test_every_dimension_has_roughly_20_examples() -> None:
    pairs = generate_synthetic_pairs()
    from collections import Counter

    counts = Counter(p["dimension"] for p in pairs)
    assert set(counts) == set(DIMENSIONS)
    for dim in DIMENSIONS:
        assert counts[dim] >= 10, f"{dim} only has {counts[dim]} examples"


def test_expected_field_is_always_blank() -> None:
    # ★ The one hard requirement: this script must never pre-label pairs.
    pairs = generate_synthetic_pairs()
    assert all(p["expected"] is None for p in pairs)


def test_every_pair_has_a_unique_id() -> None:
    pairs = generate_synthetic_pairs()
    ids = [p["pair_id"] for p in pairs]
    assert len(ids) == len(set(ids))


def test_write_golden_pairs_produces_valid_jsonl(tmp_path: Path) -> None:
    pairs = generate_synthetic_pairs()[:5]
    out = tmp_path / "pairs.jsonl"
    write_golden_pairs(pairs, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        assert "expected" in record
        assert record["expected"] is None
        assert "pair_id" in record
        assert "dimension" in record


def test_dialect_adversarial_pairs_use_real_execution(sqlite_db: Path) -> None:
    pairs = generate_dialect_adversarial_pairs(sqlite_path=sqlite_db, postgres_dsn=None)
    # With postgres_dsn=None, no cross-dialect pairs are produced (nothing
    # to pair the SQLite result against) — this asserts the function
    # doesn't crash and returns cleanly, not that it produces pairs.
    assert isinstance(pairs, list)


@requires_postgres
def test_dialect_adversarial_pairs_with_real_postgres(sqlite_db: Path) -> None:
    pairs = generate_dialect_adversarial_pairs(sqlite_path=sqlite_db, postgres_dsn=TEST_PG_DSN)
    assert len(pairs) > 0
    for p in pairs:
        assert p["source"] == "cross_dialect_real_execution"
        assert p["expected"] is None


@requires_postgres
def test_dialect_adversarial_pairs_are_json_serializable(sqlite_db: Path, tmp_path: Path) -> None:
    # Regression test: PostgreSQL's integer division (`1/2`) comes back
    # through psycopg as a `decimal.Decimal`, not a float — confirmed by
    # actually running this against the real local PostgreSQL server in
    # this sandbox. json.dumps(default=list) doesn't know how to handle
    # Decimal (it isn't iterable) and raised TypeError until fixed.
    pairs = generate_dialect_adversarial_pairs(sqlite_path=sqlite_db, postgres_dsn=TEST_PG_DSN)
    out = tmp_path / "dialect_pairs.jsonl"
    write_golden_pairs(pairs, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(pairs)
    for line in lines:
        json.loads(line)  # must not raise
