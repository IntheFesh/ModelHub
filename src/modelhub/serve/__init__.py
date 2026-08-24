"""Model serving layer: vLLM process management, prompt/schema formatting.

Public API: `build_prompt`/`make_prompt_builder`/`PromptConfig` (prompt.py,
the concrete `Callable[[NormalizedSample], str]` eval/runner.py's
`run_eval` needs); `introspect_sqlite_schema`/`render_schema_text`
(schema_format.py); `detect_gdn_kernel_status`/`diagnose_gdn_kernels`
(kernel_status.py, CLAUDE.md §5.3 GDN fast-kernel detection);
`ModelProfile`/`kv_bytes_per_token`/`load_model_profile` (model_profile.py,
the KV-cache economics MODEL-SELECTION-FINAL.md's table is derived from);
`parse_vllm_startup_log`/`VllmStartupFacts` (vllm_log_parser.py).

Actually starting a vLLM process is out of scope for this module (and for
this sandbox, which has no GPU) — that is A8/A10's job. This layer only
owns what a served model needs (the prompt/schema it's fed) and what a
served run needs to record about itself (kernel status, KV facts parsed
from its own log).
"""

from modelhub.serve.kernel_status import (
    KernelCheckResult,
    check_causal_conv1d,
    check_fla,
    detect_gdn_kernel_status,
    diagnose_gdn_kernels,
)
from modelhub.serve.model_profile import (
    ModelProfile,
    gpu_blocks_to_cacheable_tokens,
    kv_bytes_per_token,
    kv_kib_per_token,
    load_model_profile,
)
from modelhub.serve.prompt import PromptConfig, SchemaCache, build_prompt, make_prompt_builder
from modelhub.serve.schema_format import (
    ColumnSchema,
    DbSchema,
    ForeignKey,
    SchemaStyle,
    TableSchema,
    introspect_sqlite_schema,
    render_schema_text,
)
from modelhub.serve.vllm_log_parser import VllmStartupFacts, parse_vllm_startup_log

__all__ = [
    "ColumnSchema",
    "DbSchema",
    "ForeignKey",
    "KernelCheckResult",
    "ModelProfile",
    "PromptConfig",
    "SchemaCache",
    "SchemaStyle",
    "TableSchema",
    "VllmStartupFacts",
    "build_prompt",
    "check_causal_conv1d",
    "check_fla",
    "detect_gdn_kernel_status",
    "diagnose_gdn_kernels",
    "gpu_blocks_to_cacheable_tokens",
    "introspect_sqlite_schema",
    "kv_bytes_per_token",
    "kv_kib_per_token",
    "load_model_profile",
    "make_prompt_builder",
    "parse_vllm_startup_log",
    "render_schema_text",
]
