"""GDN fast-kernel effective-detection (CLAUDE.md §5.3).

Qwen3.5's GDN (gated delta net) layers silently fall back to a slow,
more memory-hungry pure-PyTorch implementation when `causal_conv1d` or
`fla` (flash-linear-attention) is missing or broken — no error, no
warning. `preflight.py`'s G1/G2 checks (this project's pre-existing,
human-authored reference) already establish the only trustworthy way to
detect this: actually invoke the real CUDA kernel on a tiny real tensor
and check the output is finite, not just `import` the package. This
module is that same check, refactored into an importable function so
serve-time manifest-writing code (and A9's gate, which reads
`kernel_status.degraded`) can call it instead of only a standalone CLI
script.

CLAUDE.md §1.3 (★ 未测 ≠ 通过): "not installed" (SKIP) and "installed but
broken" (FAIL) are both `degraded=True` — only a real, successful kernel
invocation counts as available. This sandbox has no CUDA device, so every
test of this module exercises the honest SKIP path for real, not a stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.common.capability import CheckStatus, is_available
from modelhub.common.run_manifest import KernelStatus


@dataclass(frozen=True)
class KernelCheckResult:
    status: CheckStatus
    detail: str
    error: str | None = None


def check_causal_conv1d() -> KernelCheckResult:
    """Mirrors preflight.py G1: run causal_conv1d's real CUDA kernel."""
    try:
        import torch
    except ImportError as e:
        return KernelCheckResult(
            CheckStatus.SKIP, "torch not installed", f"{type(e).__name__}: {e}"
        )

    if not torch.cuda.is_available():
        return KernelCheckResult(CheckStatus.SKIP, "no CUDA device available")

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError as e:
        return KernelCheckResult(
            CheckStatus.SKIP, "causal_conv1d not installed", f"{type(e).__name__}: {e}"
        )

    try:
        b, d, seq, k = 2, 128, 64, 4
        x = torch.randn(b, d, seq, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(d, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        out = causal_conv1d_fn(x, w, bias, activation="silu")
        torch.cuda.synchronize()
    except Exception as e:
        # capability probe: classify the failure, don't propagate it (see
        # module docstring) — one broken dependency must not crash the check.
        return KernelCheckResult(
            CheckStatus.FAIL, "kernel invocation raised", f"{type(e).__name__}: {e}"
        )

    if not bool(torch.isfinite(out).all()):
        return KernelCheckResult(
            CheckStatus.FAIL, f"kernel produced non-finite output {tuple(out.shape)}"
        )
    return KernelCheckResult(
        CheckStatus.PASS, f"real CUDA kernel executed, output shape {tuple(out.shape)}"
    )


_FLA_KERNEL_ENTRY_POINTS = (
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.delta_rule", "chunk_delta_rule"),
)


def check_fla() -> KernelCheckResult:
    """Mirrors preflight.py G2: run fla's real GDN kernel."""
    try:
        import torch
    except ImportError as e:
        return KernelCheckResult(
            CheckStatus.SKIP, "torch not installed", f"{type(e).__name__}: {e}"
        )

    if not torch.cuda.is_available():
        return KernelCheckResult(CheckStatus.SKIP, "no CUDA device available")

    try:
        import fla
    except ImportError as e:
        return KernelCheckResult(CheckStatus.SKIP, "fla not installed", f"{type(e).__name__}: {e}")

    fla_version = str(getattr(fla, "__version__", "?"))
    fn = None
    entry_point = ""
    for module_path, fn_name in _FLA_KERNEL_ENTRY_POINTS:
        try:
            fn = getattr(__import__(module_path, fromlist=[fn_name]), fn_name)
            entry_point = f"{module_path}.{fn_name}"
            break
        except (ImportError, AttributeError):
            continue
    if fn is None:
        checked = [f"{m}.{n}" for m, n in _FLA_KERNEL_ENTRY_POINTS]
        return KernelCheckResult(
            CheckStatus.WARN,
            f"fla {fla_version} installed but no known GDN kernel entry point found "
            f"(checked {checked}, API may have been renamed)",
        )

    try:
        b, t, h, k_dim, v_dim = 1, 64, 4, 64, 64
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        out = fn(
            q=torch.randn(b, t, h, k_dim, **kw),
            k=torch.randn(b, t, h, k_dim, **kw),
            v=torch.randn(b, t, h, v_dim, **kw),
            g=torch.rand(b, t, h, device="cuda", dtype=torch.float32).log(),
            beta=torch.rand(b, t, h, **kw),
        )
        o = out[0] if isinstance(out, tuple | list) else out
        torch.cuda.synchronize()
    except Exception as e:
        return KernelCheckResult(
            CheckStatus.FAIL, "kernel invocation raised", f"{type(e).__name__}: {e}"
        )

    if not bool(torch.isfinite(o).all()):
        return KernelCheckResult(CheckStatus.FAIL, f"{entry_point} produced non-finite output")
    return KernelCheckResult(CheckStatus.PASS, f"{entry_point} executed, fla {fla_version}")


def diagnose_gdn_kernels() -> dict[str, KernelCheckResult]:
    """Per-check detail, for logs/CLI output — see `detect_gdn_kernel_status`
    for the folded `KernelStatus` a run manifest actually stores."""
    return {"causal_conv1d": check_causal_conv1d(), "fla": check_fla()}


def detect_gdn_kernel_status() -> KernelStatus:
    """The `RunManifest.kernel_status` value for the current process.

    `degraded=True` whenever either kernel is not a confirmed PASS —
    SKIP (not installed / no CUDA) and WARN/FAIL (installed but not
    verifiably working) both mean "cannot trust the fast path", per the
    whitelist rule in `common/capability.py`.
    """
    diagnostics = diagnose_gdn_kernels()
    causal_conv1d_ok = is_available(diagnostics["causal_conv1d"].status)
    fla_ok = is_available(diagnostics["fla"].status)
    return KernelStatus(
        causal_conv1d=causal_conv1d_ok,
        fla=fla_ok,
        degraded=not (causal_conv1d_ok and fla_ok),
    )


__all__ = [
    "KernelCheckResult",
    "check_causal_conv1d",
    "check_fla",
    "detect_gdn_kernel_status",
    "diagnose_gdn_kernels",
]
