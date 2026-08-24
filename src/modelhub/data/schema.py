"""The canonical normalized sample schema every data source is mapped into.

BIRD's difficulty vocabulary (simple/moderate/challenging) and Spider's
(easy/medium/hard/extra) are NOT on a comparable scale — a BIRD "moderate"
and a Spider "medium" are not the same amount of query complexity, they
come from two different labeling methodologies. This schema keeps each
sample's difficulty string as-is (source-scoped), rather than force-mapping
both onto one shared ordinal, so downstream stratified reporting never
silently implies a cross-dataset comparison that isn't there.
"""

from __future__ import annotations

from enum import StrEnum

from modelhub.common.config import ModelHubBaseConfig


class Source(StrEnum):
    BIRD = "bird"
    SPIDER = "spider"
    MINIDEV_SELECT = "minidev_select"  # Mini-Dev V2's 500 SELECT-only (quick-eval set)
    MINIDEV_CRUD = "minidev_crud"  # Mini-Dev V2's 270 CRUD (safety-gate adversarial set)


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Dialect(StrEnum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class NormalizedSample(ModelHubBaseConfig):
    """One text-to-SQL example, in the schema every downstream module consumes.

    ``sample_id`` is stable and globally unique: ``{source}:{split}:{original_id}``
    (optionally suffixed with ``:{dialect}`` for Mini-Dev's multi-dialect
    variants), never a bare integer that could collide across sources.
    """

    sample_id: str
    db_id: str
    question: str
    evidence: str | None  # BIRD-only external knowledge hint; None elsewhere
    gold_sql: str
    difficulty: str | None  # source-scoped vocabulary; None if not labeled
    source: Source
    split: Split
    dialect: Dialect = Dialect.SQLITE
