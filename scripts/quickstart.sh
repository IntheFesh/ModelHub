#!/usr/bin/env bash
# `make quickstart` — one-command environment setup for this repo.
#
# What this does, in order (every step is idempotent — safe to re-run):
#   1. Check Python >= 3.11 is available.
#   2. Create .venv/ and install the repo in editable mode with the
#      dev/data/serve/gateway/monitor/sqlexec extras (same as `make install`).
#   3. Best-effort start a local Redis and PostgreSQL (see
#      scripts/lib_dev_services.sh) — these unlock `make demo` and the
#      handful of tests that are otherwise honestly SKIPPED (never faked)
#      when no local instance is reachable.
#   4. Run ruff + mypy --strict + the full test suite as a real, end-to-end
#      proof the environment actually works — not just "install finished
#      with no errors".
#
# What this deliberately does NOT do: install CUDA/GPU packages (the
# `train` extra), download any model weights, or start vLLM/veRL — this
# sandbox-friendly setup only prepares the CPU-only, no-GPU parts of the
# platform that this repo can honestly run without real hardware
# (CLAUDE.md §0's "所有对外声称的数字必须能追溯到一次真实 run 的落盘
# 产物" — this script proves what it claims by actually running it, it
# does not claim more than that).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck source=./lib_dev_services.sh
source scripts/lib_dev_services.sh

echo "════════════════════════════════════════════════════════════"
echo " ModelHub quickstart"
echo "════════════════════════════════════════════════════════════"

echo
echo "[1/4] Python version"
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "  ERROR: $PYTHON_BIN not found. Install Python 3.11+ first." >&2
    exit 1
fi
PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_OK=$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')
if [ "$PY_OK" != "1" ]; then
    echo "  ERROR: found Python $PY_VERSION, this project requires >= 3.11." >&2
    exit 1
fi
echo "  $PYTHON_BIN -> $PY_VERSION OK"

echo
echo "[2/4] Virtualenv + editable install"
if [ ! -x .venv/bin/python ]; then
    echo "  creating .venv/ ..."
    "$PYTHON_BIN" -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
else
    echo "  .venv/ already exists, reusing it"
fi
echo "  installing modelhub[dev,data,serve,gateway,monitor,sqlexec] (this can take a minute the first time)..."
.venv/bin/pip install -e ".[dev,data,serve,gateway,monitor,sqlexec]" --quiet
echo "  done"

echo
echo "[3/4] Local dev services (best-effort, optional)"
ensure_redis || true
ensure_postgres || true

echo
echo "[4/4] Verifying the environment actually works"
echo "  -- ruff --"
.venv/bin/ruff check src tests scripts
echo "  -- mypy --strict src/modelhub --"
.venv/bin/mypy src/modelhub
echo "  -- pytest (excludes requires_gpu/requires_network) --"
.venv/bin/pytest -q -m "not requires_gpu and not requires_network"

echo
echo "════════════════════════════════════════════════════════════"
echo " Ready. Suggested next steps:"
echo "════════════════════════════════════════════════════════════"
echo "   make showcase          # run the 3 real, GPU-free end-to-end demos"
echo "   make demo              # gateway request -> sqlexec -> compare -> gate REJECT -> incident log"
echo "   make bad-model-drill   # B3: 3 synthetic bad checkpoints, real admission-gate REJECT verdicts"
echo "   make night-queue       # unattended overnight task orchestration"
echo "   make verify-a0         # per-round acceptance, e.g. A0 (swap a0 for any round id)"
echo "   make help              # full command list"
