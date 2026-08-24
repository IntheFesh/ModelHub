"""Unit tests for serve/vllm_log_parser.py.

The log snippets below are hand-written to match vLLM's known, stable
log line formats (see module docstring for which patterns are confirmed
vs. best-effort) — they are not captured from a real run (no GPU in this
sandbox to produce one).
"""

from __future__ import annotations

from modelhub.serve.vllm_log_parser import parse_vllm_startup_log

_REALISTIC_LOG = """\
INFO 08-24 04:00:00 model_runner.py:915] Starting to load model qwen3.5-9b...
INFO 08-24 04:00:03 model_runner.py:920] Loading model weights took 15.2334 GB
INFO 08-24 04:00:05 gpu_executor.py:122] # GPU blocks: 7449, # CPU blocks: 2048
INFO 08-24 04:00:05 worker.py:232] Maximum concurrency for 2048 tokens per request: 21.63x
INFO 08-24 04:00:06 llm_engine.py:400] Engine ready.
"""


def test_parses_all_known_fields_from_a_realistic_log() -> None:
    facts = parse_vllm_startup_log(_REALISTIC_LOG)
    assert facts.gpu_blocks == 7449
    assert facts.cpu_blocks == 2048
    assert facts.weight_load_gb == 15.2334
    assert facts.max_concurrency_tokens == 2048
    assert facts.max_concurrency == 21.63
    assert len(facts.matched_lines) == 3


def test_empty_log_yields_all_none() -> None:
    facts = parse_vllm_startup_log("")
    assert facts.gpu_blocks is None
    assert facts.cpu_blocks is None
    assert facts.weight_load_gb is None
    assert facts.max_concurrency is None
    assert facts.gdn_state_footprint_bytes is None
    assert facts.matched_lines == ()


def test_irrelevant_log_text_yields_all_none() -> None:
    facts = parse_vllm_startup_log("INFO some totally unrelated log line\nWARNING another one\n")
    assert facts.gpu_blocks is None
    assert facts.matched_lines == ()


def test_gdn_state_pattern_best_effort_match() -> None:
    log = _REALISTIC_LOG + "INFO 08-24 04:00:04 gdn_cache.py:50] GDN recurrent state: 18.0 MiB\n"
    facts = parse_vllm_startup_log(log)
    assert facts.gdn_state_footprint_bytes == int(18.0 * 1024 * 1024)


def test_gdn_state_absent_is_none_not_zero() -> None:
    facts = parse_vllm_startup_log(_REALISTIC_LOG)
    assert facts.gdn_state_footprint_bytes is None


def test_partial_log_only_gpu_blocks_present() -> None:
    facts = parse_vllm_startup_log("INFO x] # GPU blocks: 100, # CPU blocks: 20\n")
    assert facts.gpu_blocks == 100
    assert facts.cpu_blocks == 20
    assert facts.weight_load_gb is None
    assert facts.max_concurrency is None
