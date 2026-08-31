#!/usr/bin/env bash
# `make showcase` — runs, back to back, the three scripts in this repo
# that are genuinely real, GPU-free, end-to-end demonstrations (not mocks
# standing in for the real thing — see each script's own module
# docstring for exactly which piece, if any, is a stand-in):
#
#   1. scripts/demo.py            gateway request -> sqlexec -> compare
#                                  -> Prometheus scrape -> admission-gate
#                                  REJECT -> incident-log append
#   2. scripts/bad_model_drill.py three synthetic "bad" checkpoints run
#                                  through the real 5-gate admission
#                                  check, each producing a real REJECT
#   3. scripts/night_queue.py     unattended overnight task orchestration
#                                  (risk-ascending queue, watchdog, real
#                                  heartbeat + manifests)
#
# Only the "model" itself is ever a stand-in (a fixed-response stub —
# this sandbox has no GPU to serve a real one); everything downstream of
# that is real project code exercised for real, per CLAUDE.md's anti-
# cheat rules.
#
# ★ Re-running this appends new, real entries to docs/incident-log.md
# and writes new artifacts under artifacts/ (gitignored) — that is
# expected, not a bug: these are genuine append-only logs, the same way
# a real production system's incident log only grows.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -x .venv/bin/python ]; then
    echo "No .venv/ found — run 'make quickstart' (or 'bash scripts/quickstart.sh') first." >&2
    exit 1
fi

# shellcheck source=./lib_dev_services.sh
source scripts/lib_dev_services.sh

echo "════════════════════════════════════════════════════════════"
echo " ModelHub showcase — 3 real, GPU-free end-to-end demos"
echo "════════════════════════════════════════════════════════════"

echo
echo "[services] making sure Redis is up (needed by scripts/demo.py's gateway)..."
ensure_redis || echo "  Redis unavailable — scripts/demo.py will fail to connect; the other two demos don't need it."

echo
echo "──────────────────────────────────────────────────────────────"
echo " 1/3  scripts/demo.py"
echo "──────────────────────────────────────────────────────────────"
.venv/bin/python scripts/demo.py

echo
echo "──────────────────────────────────────────────────────────────"
echo " 2/3  scripts/bad_model_drill.py"
echo "──────────────────────────────────────────────────────────────"
.venv/bin/python scripts/bad_model_drill.py

echo
echo "──────────────────────────────────────────────────────────────"
echo " 3/3  scripts/night_queue.py"
echo "──────────────────────────────────────────────────────────────"
.venv/bin/python scripts/night_queue.py

echo
echo "════════════════════════════════════════════════════════════"
echo " Showcase complete. Real artifacts worth looking at:"
echo "════════════════════════════════════════════════════════════"
echo "   docs/incident-log.md      (append-only, git-tracked)"
echo "   artifacts/bad_models/README.md"
echo "   artifacts/queues/<today>*.json"
