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


def scan(paths: list[Path]) -> tuple[list[tuple[str, int, str]], list[str]]:
    """Returns (hits, unreadable_files). A file this scan cannot even read
    must never silently count as "0 placeholders found" — that is the
    same skip-counted-as-pass pattern CLAUDE.md §1.3 forbids at the
    dependency-probe level, just showing up at file-discovery time
    instead; the caller decides how to report `unreadable_files`, but it
    must never be dropped on the floor."""
    hits: list[tuple[str, int, str]] = []
    unreadable: list[str] = []
    for f in _iter_files(paths):
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unreadable.append(str(f))
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _PLACEHOLDER_PATTERN.finditer(line):
                hits.append((str(f), lineno, m.group(0)))
    return hits, unreadable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)

    hits, unreadable = scan(args.paths)

    if unreadable:
        print(f"check_placeholders: {len(unreadable)} file(s) could not be read\n", file=sys.stderr)
        for path in unreadable:
            print(f"  unreadable: {path}", file=sys.stderr)
        print(file=sys.stderr)

    if hits:
        print(f"check_placeholders: {len(hits)} placeholder(s) found\n", file=sys.stderr)
        for path, lineno, token in hits:
            print(f"{path}:{lineno}: {token}", file=sys.stderr)
        return 1
    if unreadable:
        # an unreadable file inside a scanned path is itself a reason not
        # to report "clean" — we genuinely don't know if it contains a
        # placeholder.
        return 1
    print("check_placeholders: clean (0 placeholders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
