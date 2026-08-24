"""Shared pytest fixtures/config.

`scripts/` holds standalone CLI tools (check_no_cheating.py,
check_placeholders.py, night_queue.py, ...) that are not part of the
`modelhub` package. Tests that exercise them import by module name, so put
`scripts/` on `sys.path` once, here, instead of every test file doing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
