"""Content hashing for dedup keys, split-integrity checks, and dataset versioning."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from modelhub.data.schema import NormalizedSample

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_for_hash(text: str) -> str:
    """Collapse whitespace and lowercase before hashing.

    Two questions/SQL strings that differ only in incidental whitespace or
    casing are still the "same" for dedup purposes — hashing the raw bytes
    would undercount duplicates that come from re-scraping or re-formatting
    the same underlying pair.
    """
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def question_hash(question: str) -> str:
    normalized = _normalize_text_for_hash(question)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sql_hash(sql: str) -> str:
    normalized = _normalize_text_for_hash(sql)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dataset_content_hash(samples: Sequence[NormalizedSample]) -> str:
    """Deterministic hash of a dataset version's full sample set.

    Order-independent (sorted by sample_id first) so shuffling the input
    list doesn't change the hash — only the actual content does.
    """
    ordered = sorted(samples, key=lambda s: s.sample_id)
    parts = [
        f"{s.sample_id}\x1f{s.db_id}\x1f{question_hash(s.question)}\x1f{sql_hash(s.gold_sql)}"
        for s in ordered
    ]
    canonical = "\x1e".join(parts)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
