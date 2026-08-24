#!/usr/bin/env python3
"""★★★ DEMO STUB — NOT A REAL MODEL. ★★★

A tiny FastAPI server exposing the OpenAI-completions-shaped endpoint
`eval/model_client.py::HttpModelClient` expects, so `scripts/demo.py`
can drive the REAL gateway (`gateway/app.py`) and the REAL sqlexec/
compare code paths end to end without a GPU (this sandbox has none —
see docs/design-decisions.md). Its "generated" SQL is one fixed,
hardcoded string; its token counts are a cheap character-count
estimate, not a real tokenizer. Nothing this server returns is ever
presented anywhere in this project as a real inference result,
benchmark number, or accuracy claim — the demo it backs is a tour of
the OPERATIONAL flow (auth/rate-limit/routing/circuit-breaker/billing/
monitoring/admission-gate/incident-log), not a performance claim.

Lives outside `src/modelhub/` deliberately: CLAUDE.md §1.4 confines
mocks to `tests/` (never imported by `src/`) — this is the same rule
applied to a demo-only stub, which belongs in `scripts/` for the same
reason, not inside the real package.
"""

from __future__ import annotations

import argparse
import sys

from fastapi import FastAPI
from pydantic import BaseModel

STUB_SQL = "SELECT id, name FROM students WHERE id = 1"

app = FastAPI(title="ModelHub Demo Stub Model Server — NOT A REAL MODEL")


class _CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 512


@app.post("/v1/completions")
def completions(req: _CompletionRequest) -> dict[str, object]:
    prompt_tokens = max(len(req.prompt) // 4, 1)
    completion_tokens = max(len(STUB_SQL) // 4, 1)
    return {
        "model": req.model,
        "choices": [{"text": STUB_SQL, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "warning": "THIS IS A DEMO STUB, NOT A REAL MODEL"}


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
