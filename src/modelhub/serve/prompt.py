"""Text2SQL prompt construction — the concrete `prompt_builder` callable
`eval/runner.py::run_eval` was deliberately left generic over (A4/DD-0012:
`run_eval` itself must not know dataset-specific facts; the eval layer
only needs *a* `Callable[[NormalizedSample], str]`, and this module is
where that callable actually gets built).

Prompt template text is a `PromptConfig` field with a real default, not a
string literal scattered through the render function — CLAUDE.md §4's
"hyperparameters live in configs/, not magic numbers in code" applies to
prompt templates exactly the same way it applies to `lr`/`batch_size`:
swapping the instruction wording is a config change, not a code change.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.schema import NormalizedSample
from modelhub.serve.schema_format import (
    DbSchema,
    SchemaStyle,
    introspect_sqlite_schema,
    render_schema_text,
)

_DEFAULT_INSTRUCTION = (
    "You are a SQLite expert. Given a database schema and a question, "
    "write a single SQLite SQL query that answers the question. "
    "Output only the SQL query, with no explanation."
)


class PromptConfig(ModelHubBaseConfig):
    instruction: str = _DEFAULT_INSTRUCTION
    schema_style: SchemaStyle = "ddl"
    include_evidence: bool = True


def build_prompt(sample: NormalizedSample, schema_text: str, config: PromptConfig) -> str:
    parts = [config.instruction, "", "### Database Schema", schema_text]
    if config.include_evidence and sample.evidence:
        parts += ["", "### Evidence", sample.evidence]
    parts += ["", "### Question", sample.question, "", "### SQL"]
    return "\n".join(parts)


class SchemaCache:
    """Per-db_id schema-introspection + rendering cache.

    Evaluating N samples against the same handful of databases (BIRD dev
    has ~500 samples over ~11 databases) must not re-run PRAGMA
    introspection and re-render the schema text N times over — the schema
    itself cannot change mid-eval-run (the db files are read-only
    mounts, CLAUDE.md §7), so this cache has no invalidation logic, unlike
    `eval/gold_cache.py`'s content-hash-keyed cache (which exists
    specifically because that content *can* legitimately change between
    runs).
    """

    def __init__(self, db_root: Path, config: PromptConfig) -> None:
        self._db_root = db_root
        self._config = config
        self._schema_text: dict[str, str] = {}

    def get(self, db_id: str) -> str:
        if db_id not in self._schema_text:
            db_path = self._db_root / db_id / f"{db_id}.sqlite"
            schema: DbSchema = introspect_sqlite_schema(db_path, db_id=db_id)
            self._schema_text[db_id] = render_schema_text(schema, style=self._config.schema_style)
        return self._schema_text[db_id]


def make_prompt_builder(
    db_root: Path, config: PromptConfig | None = None
) -> Callable[[NormalizedSample], str]:
    """Build the `prompt_builder` callable `eval.run_eval` expects."""
    resolved_config = config or PromptConfig()
    cache = SchemaCache(db_root, resolved_config)

    def prompt_builder(sample: NormalizedSample) -> str:
        schema_text = cache.get(sample.db_id)
        return build_prompt(sample, schema_text, resolved_config)

    return prompt_builder


__all__ = [
    "PromptConfig",
    "SchemaCache",
    "build_prompt",
    "make_prompt_builder",
]
