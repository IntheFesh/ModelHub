"""DB schema introspection and deterministic prompt-ready rendering.

Real `sqlite3` introspection (`PRAGMA table_info`/`PRAGMA foreign_key_list`
against the actual db file), not a schema cached from BIRD/Spider's JSON
metadata — the prompt the model sees must match the database sqlexec will
actually execute the model's SQL against, and those two can drift (a BIRD
`tables.json` entry does not always describe every column truthfully).

Table and column order is always the db file's own declaration order
(`sqlite_master`/`PRAGMA table_info` order, not sorted) — CREATE TABLE
order is itself informative for a model trying to infer intended reading
order (e.g. natural id-then-attributes patterns), and re-sorting it would
throw that signal away for no benefit (rendering is already deterministic
because a single db file's declaration order never changes between runs).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_SQLITE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type: str
    not_null: bool
    is_primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSchema, ...]


@dataclass(frozen=True)
class DbSchema:
    db_id: str
    tables: tuple[TableSchema, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)

    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)


def introspect_sqlite_schema(db_path: Path, *, db_id: str) -> DbSchema:
    """Read `db_path`'s real schema via SQLite's own PRAGMA introspection.

    Read-only PRAGMA/catalog queries only — this is the harness reading
    its own database's structure, not user/model-controlled SQL, so it
    does not go through sqlexec's sandboxed `execute_isolated` (same
    reasoning as `eval/gold_cache.py`'s direct file-hash read).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=_SQLITE_TIMEOUT_S)
    try:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            ).fetchall()
        ]

        tables = []
        for table_name in table_names:
            columns = tuple(
                ColumnSchema(
                    name=row[1],
                    type=row[2],
                    not_null=bool(row[3]),
                    is_primary_key=bool(row[5]),
                )
                for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            )
            tables.append(TableSchema(name=table_name, columns=columns))

        foreign_keys = []
        for table_name in table_names:
            for row in conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall():
                # PRAGMA foreign_key_list columns: id, seq, table, from, to, ...
                foreign_keys.append(
                    ForeignKey(
                        from_table=table_name,
                        from_column=row[3],
                        to_table=row[2],
                        to_column=row[4],
                    )
                )

        return DbSchema(db_id=db_id, tables=tuple(tables), foreign_keys=tuple(foreign_keys))
    finally:
        conn.close()


SchemaStyle = Literal["ddl", "compact"]


def render_schema_text(
    schema: DbSchema,
    *,
    style: SchemaStyle = "ddl",
    sample_rows: Mapping[str, list[tuple[object, ...]]] | None = None,
) -> str:
    """Render `schema` into prompt-ready text.

    `style="ddl"`: a `CREATE TABLE` block per table plus a trailing
    foreign-key summary — closest to what the model will have seen in
    pretraining, most information-dense.
    `style="compact"`: `table(col1, col2, ...)` one-liners — shorter,
    useful for the prompt-length sweep A8 runs (A6's routing threshold is
    keyed on schema token length, so having a deliberately terser
    alternative rendering matters, not just a nicer-looking one).

    `sample_rows`, if given, appends up to the rows provided per table as
    `-- e.g. (...)` comments (BIRD's "evidence" convention) — never
    fetched by this function itself, so a caller controls exactly how
    many rows (if any) leak into the prompt.
    """
    if style == "ddl":
        return _render_ddl(schema, sample_rows)
    if style == "compact":
        return _render_compact(schema)
    raise ValueError(f"unknown schema style: {style!r}")


def _render_ddl(
    schema: DbSchema, sample_rows: Mapping[str, list[tuple[object, ...]]] | None
) -> str:
    fks_by_table: dict[str, list[ForeignKey]] = {}
    for fk in schema.foreign_keys:
        fks_by_table.setdefault(fk.from_table, []).append(fk)

    blocks = []
    for table in schema.tables:
        lines = [f"CREATE TABLE {table.name} ("]
        col_lines = []
        for col in table.columns:
            parts = [f"  {col.name} {col.type}"]
            if col.is_primary_key:
                parts.append("PRIMARY KEY")
            if col.not_null and not col.is_primary_key:
                parts.append("NOT NULL")
            col_lines.append(" ".join(parts))
        for fk in fks_by_table.get(table.name, ()):
            col_lines.append(
                f"  FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})"
            )
        lines.append(",\n".join(col_lines))
        lines.append(");")
        block = "\n".join(lines)
        if sample_rows and table.name in sample_rows and sample_rows[table.name]:
            example = sample_rows[table.name][0]
            block += f"\n-- e.g. {example}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _render_compact(schema: DbSchema) -> str:
    lines = []
    for table in schema.tables:
        col_names = ", ".join(c.name for c in table.columns)
        lines.append(f"{table.name}({col_names})")
    if schema.foreign_keys:
        lines.append("-- foreign keys:")
        for fk in schema.foreign_keys:
            lines.append(f"--   {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
    return "\n".join(lines)
