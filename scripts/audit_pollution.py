#!/usr/bin/env python3
"""Audit `artifacts/runs/*/manifest.json` for CLAUDE.md §3.1's three
pollution flags (git_dirty / contaminated / degraded) — PLAN.md A12 item
3: "校验所有被引用的 run 三个污染位均为 false，有问题的列出来".

Reuses `render_docs.py::discover_runs` (same scan, same split) rather
than re-walking `artifacts/runs/` a second time with slightly different
logic — this script's only job on top of that is a human-readable
report and a non-zero exit code when a polluted run exists, for
`make verify-a12`/CI to fail on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from render_docs import discover_runs


def render_report(artifacts_root: Path) -> tuple[str, bool]:
    """Returns (report_text, all_clean)."""
    runs = discover_runs(artifacts_root)
    lines = [
        f"audit_pollution: scanned {artifacts_root}",
        f"  clean runs:    {len(runs.clean)}",
        f"  polluted runs: {len(runs.polluted)}",
    ]
    if not runs.clean and not runs.polluted:
        lines.append(
            "  (no runs found — nothing to audit; this is the honest state of an "
            "environment that has never executed a real bench/eval run)"
        )
    for m in runs.polluted:
        lines.append(f"  ✗ {m.run_id}: {', '.join(m.pollution_reasons())}")
    for m in runs.clean:
        lines.append(f"  ✓ {m.run_id}: clean")
    return "\n".join(lines), not runs.polluted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args(argv)

    report, all_clean = render_report(args.artifacts_root)
    print(report)
    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
