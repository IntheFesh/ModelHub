#!/usr/bin/env python3
"""Scan for machine-checkable placeholder markers (CLAUDE.md §3.3).

`[X]` / `[TODO]` / `[待测]` must be zero before anything ships. This is
intentionally narrow (bracketed placeholder tokens, not every `TODO`
comment in source — those are normal in-flight development markers, not
the "looks-finished-but-isn't" placeholders this check targets) and is run
at the end of every round (`make check-placeholders`), required to be zero
before delivery.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PLACEHOLDER_PATTERN = re.compile(r"\[X\]|\[TODO\]|\[待测\]", re.IGNORECASE)
_SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".txt"}
_SKIP_DIR_NAMES = {".venv", ".git", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}


def _iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if root.is_file():
            files.append(root)
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            if any(part in _SKIP_DIR_NAMES for part in f.parts):
                continue
            if f.suffix in _SCAN_SUFFIXES:
                files.append(f)
    return files


def scan(paths: list[Path]) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for f in _iter_files(paths):
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _PLACEHOLDER_PATTERN.finditer(line):
                hits.append((str(f), lineno, m.group(0)))
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)

    hits = scan(args.paths)
    if hits:
        print(f"check_placeholders: {len(hits)} placeholder(s) found\n", file=sys.stderr)
        for path, lineno, token in hits:
            print(f"{path}:{lineno}: {token}", file=sys.stderr)
        return 1
    print("check_placeholders: clean (0 placeholders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
