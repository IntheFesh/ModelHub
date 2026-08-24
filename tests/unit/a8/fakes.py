"""Fake `ModelClient` for bench/ tests, with configurable latency and
failure injection — CLAUDE.md §1.4: mocks live only in `tests/` and are
never imported by `src/`. This is a real implementation of the
`ModelClient` Protocol (same interface `tests/unit/a4/fakes.py` fakes),
just with two extra knobs `bench/load_test.py`'s tests need that A4's
fakes don't: an actual (small, real) per-call delay to prove concurrency
genuinely parallelizes, and deterministic failure injection to exercise
`LoadTestResult.error_breakdown`.
"""

from __future__ import annotations

import threading
import time

from modelhub.eval.model_client import GenerationResult


class BenchFakeClient:
    def __init__(
        self,
        *,
        latency_s: float = 0.0,
        fail_every_n: int | None = None,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        self._latency_s = latency_s
        self._fail_every_n = fail_every_n
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._call_count = 0
        self._lock = threading.Lock()

    def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, timeout_s: float
    ) -> GenerationResult:
        with self._lock:
            self._call_count += 1
            call_number = self._call_count

        if self._latency_s:
            time.sleep(self._latency_s)

        if self._fail_every_n and call_number % self._fail_every_n == 0:
            raise RuntimeError(f"simulated failure on call {call_number}")

        return GenerationResult(
            text="SELECT 1",
            finish_reason="stop",
            model_id="bench-fake",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )
