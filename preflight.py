#!/usr/bin/env python3
"""
ModelHub Day-0 预检 v2
======================
把计划里所有"我假设成立"的东西，在你的机器上逐条测掉。

v2 新增 G 段：Qwen3.5 GDN 混合架构专项检查。
  最关键的是 G1/G2——GDN 的快 kernel 缺失时模型会【静默回退】到慢的
  PyTorch 实现，不报错不告警，你的吞吐/显存/MFU 数字会全部错而你不知道。

设计原则（与 CLAUDE.md 一致）：
  - 每项检查独立，失败不阻断后续
  - 不吞异常：失败时打印真实的异常类型与消息
  - 不伪造通过：装不了的依赖报 SKIP 而不是 PASS
  - 每项失败都给出兜底方案

用法：
    python3 preflight.py                    # 不需要下模型的全部检查
    python3 preflight.py --quick            # 跳过算力/带宽实测
    python3 preflight.py --with-model       # 含 vLLM 实际加载 Qwen3.5-9B（要下 ~18GB）
    python3 preflight.py --json out.json

退出码：0 = 无 CRITICAL 失败；1 = 有 CRITICAL 失败
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


@dataclass
class Check:
    id: str
    name: str
    critical: bool
    status: str = SKIP
    detail: str = ""
    error: str = ""
    fallback: str = ""
    data: dict[str, Any] = field(default_factory=dict)


RESULTS: list[Check] = []


def run(check_id: str, name: str, critical: bool, fallback: str):
    def deco(fn: Callable[[Check], None]):
        c = Check(id=check_id, name=name, critical=critical, fallback=fallback)
        print(f"  [{check_id}] {name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            fn(c)
        except ImportError as e:
            c.status = SKIP
            c.error = f"{type(e).__name__}: {e}"
            c.detail = "依赖未安装"
        except Exception as e:
            c.status = FAIL
            c.error = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc(limit=3).strip().splitlines()
            c.detail = tb[-1] if tb else "未知失败"
        c.data["elapsed_s"] = round(time.time() - t0, 2)
        mark = {PASS: "✓", FAIL: "✗", SKIP: "-", WARN: "!"}[c.status]
        print(f"{mark} {c.status}  {c.detail}")
        if c.error and c.status != PASS:
            print(f"        └─ {c.error}")
        RESULTS.append(c)
        return fn

    return deco


def sh(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def section(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# ══════════════════════════════════════════════════════════════════════
# A · 系统与驱动
# ══════════════════════════════════════════════════════════════════════

def checks_system() -> None:
    section("A · 系统与驱动")

    @run("A1", "nvidia-smi 可用 / GPU 列表", True,
         "驱动未装或未加载。装 NVIDIA driver 570+ 后重试。")
    def _(c: Check):
        if not shutil.which("nvidia-smi"):
            c.status = FAIL
            c.detail = "nvidia-smi 不在 PATH"
            return
        rc, out, err = sh(["nvidia-smi",
                           "--query-gpu=index,name,memory.total,driver_version",
                           "--format=csv,noheader"])
        if rc != 0:
            c.status = FAIL
            c.detail = "nvidia-smi 返回非零"
            c.error = err[:300]
            return
        gpus = [l.strip() for l in out.splitlines() if l.strip()]
        c.data.update(gpus=gpus, gpu_count=len(gpus))
        c.status = PASS
        c.detail = f"{len(gpus)} 张 | " + " ; ".join(gpus)

    @run("A2", "是否在 WSL2（Blackwell 上强烈不推荐）", False,
         "迁到原生 Linux。WSL2 下 CUDA graph 问题多，被迫 enforce-eager "
         "吞吐会掉数倍，所有压测数字作废。")
    def _(c: Check):
        rel = platform.release().lower()
        is_wsl = "microsoft" in rel or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        c.data["kernel"] = platform.release()
        c.status = WARN if is_wsl else PASS
        c.detail = "检测到 WSL2 ← 建议迁原生 Linux" if is_wsl else f"原生 Linux ({platform.release()})"

    @run("A3", "磁盘可用 ≥ 150 GB（BIRD 33.4GB + Arctic 15GB + Qwen3.5 18GB + ckpt）", True,
         "清盘或挂外部存储。BIRD 解压后还会再涨。")
    def _(c: Check):
        target = os.path.abspath(os.environ.get("MODELHUB_DATA_DIR", "."))
        free_gb = shutil.disk_usage(target).free / 1024 ** 3
        c.data.update(path=target, free_gb=round(free_gb, 1))
        c.status = PASS if free_gb >= 150 else FAIL
        c.detail = f"{target}: 可用 {free_gb:.1f} GB"

    @run("A4", "libnvptxcompiler.so 存在（CUDA 12.8/12.9 已知缺失 → PTX JIT 全废）", False,
         "缺失会导致 cpp_extension.load() 编译失败，FlashAttention / causal_conv1d / "
         "vLLM 自定义算子都装不上。换 CUDA 版本、换镜像、或改用预编译 wheel。"
         "解决不了 → 训练线与 Qwen3.5 线整体移到 A100。")
    def _(c: Check):
        roots = ["/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-12.9",
                 "/usr/local/cuda-12.8", "/usr", "/opt/cuda"]
        found: list[str] = []
        for r in roots:
            if not os.path.isdir(r):
                continue
            for dirpath, _dn, filenames in os.walk(r):
                if dirpath.count(os.sep) - r.count(os.sep) > 3:
                    continue
                found += [os.path.join(dirpath, f) for f in filenames
                          if f.startswith("libnvptxcompiler")]
        c.data["found"] = found[:5]
        c.status = PASS if found else WARN
        c.detail = f"找到 {len(found)} 个: {found[0]}" if found else "未找到 ← PTX JIT 可能不可用"


# ══════════════════════════════════════════════════════════════════════
# B · PyTorch 与 sm_120
# ══════════════════════════════════════════════════════════════════════

def checks_torch() -> None:
    section("B · PyTorch 与 sm_120")

    @run("B1", "torch 版本 / CUDA / 编译架构表是否含本机架构", True,
         "装 torch 2.11+cu129 或更新。注意 cu130 会打断 bitsandbytes。")
    def _(c: Check):
        import torch
        if not torch.cuda.is_available():
            c.status = FAIL
            c.detail = f"torch {torch.__version__} 但 CUDA 不可用"
            return
        arch = torch.cuda.get_arch_list()
        cc = torch.cuda.get_device_capability(0)
        sm = f"sm_{cc[0]}{cc[1]}"
        has = any(sm in a for a in arch)
        c.data.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      arch_list=arch, compute_capability=list(cc), sm=sm, arch_ok=has)
        c.status = PASS if has else FAIL
        c.detail = f"torch {torch.__version__}/cu{torch.version.cuda} 设备{sm} 架构表含: {has}"
        if not has:
            c.error = f"编译架构 {arch} 不含 {sm} → 会 PTX JIT 或报 no kernel image"

    @run("B2", "bf16 矩阵乘实测（捕获 sm_120 cuBLAS 缺 kernel）", True,
         "CUBLAS_STATUS_EXECUTION_FAILED = torch 缺 sm_120 cuBLAS kernel。"
         "升 torch 2.11+cu129，并 rm -rf ~/.cache/torch /tmp/*_compiled_cache。")
    def _(c: Check):
        import torch
        if not torch.cuda.is_available():
            c.status = SKIP; c.detail = "无 CUDA"; return
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        out = (a @ a).float()
        torch.cuda.synchronize()
        c.data["out_mean"] = float(out.mean())
        c.status = PASS
        c.detail = "bf16 4096³ matmul 成功"

    @run("B3", "反向传播实测", True, "反向失败通常是 cuBLAS/cuDNN 不匹配，同 B2 解法。")
    def _(c: Check):
        import torch
        if not torch.cuda.is_available():
            c.status = SKIP; c.detail = "无 CUDA"; return
        x = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (x @ w).float().pow(2).mean().backward()
        torch.cuda.synchronize()
        ok = x.grad is not None and bool(torch.isfinite(x.grad).all())
        c.status = PASS if ok else FAIL
        c.detail = f"backward 成功, grad 有限: {ok}"

    @run("B4", "torch.compile / inductor（sm_120 上失败会伪装成 OOM）", False,
         "★ 若报 'CUDA driver error: out of memory' 但显存充足，根因是 Triton kernel "
         "在 sm_120 上启动失败，不是显存不够。设 TORCHDYNAMO_DISABLE=1。"
         "半夜撞上这个不要往显存方向查。")
    def _(c: Check):
        import torch
        if not torch.cuda.is_available():
            c.status = SKIP; c.detail = "无 CUDA"; return
        cf = torch.compile(lambda t: torch.nn.functional.gelu(t) * 2.0)
        y = cf(torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        c.data["shape"] = list(y.shape)
        c.status = PASS
        c.detail = "torch.compile 成功"

    @run("B5", "Triton 编译并启动自定义 kernel（Liger Kernel 依赖）", False,
         "Triton 不可用 → Liger Kernel 无法使用。该项 JD 线要么租卡做，"
         "要么降为认知档（讲原理，诚实说未实测）。")
    def _(c: Check):
        import torch, triton, triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            off = pid * BLOCK + tl.arange(0, BLOCK)
            m = off < n
            tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

        n = 4096
        x, y = torch.rand(n, device="cuda"), torch.rand(n, device="cuda")
        o = torch.empty_like(x)
        _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
        torch.cuda.synchronize()
        ok = bool(torch.allclose(o, x + y, atol=1e-5))
        c.data["triton_version"] = triton.__version__
        c.status = PASS if ok else FAIL
        c.detail = f"triton {triton.__version__}, kernel 结果正确: {ok}"


# ══════════════════════════════════════════════════════════════════════
# C · 训练侧依赖
# ══════════════════════════════════════════════════════════════════════

def checks_train() -> None:
    section("C · 训练侧依赖（JD 项）")

    @run("C1", "bitsandbytes 4bit 量化 + 前向（QLoRA 依赖）", False,
         "★ libnvJitLink.so.13 / cdequantize_blockwise_fp32 符号错 = bnb 编译于 cu12 "
         "而 torch 在 cu130。把 torch 钉回 cu129。仍失败 → QLoRA 移到租的 A100。")
    def _(c: Check):
        import torch, bitsandbytes as bnb
        lin = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16).cuda()
        out = lin(torch.randn(8, 512, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        c.data["bnb_version"] = getattr(bnb, "__version__", "?")
        c.status = PASS if bool(torch.isfinite(out).all()) else FAIL
        c.detail = f"bnb {c.data['bnb_version']} Linear4bit 前向正常"

    @run("C2", "FlashAttention import + 实际调用", False,
         "sm_120 上有已知 symbol error。兜底用 PyTorch SDPA，或 vLLM 侧改 FlashInfer。"
         "训练侧必须要 FA → 租卡。")
    def _(c: Check):
        import torch
        from flash_attn import flash_attn_func  # type: ignore
        q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        o = flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
        c.status = PASS if bool(torch.isfinite(o).all()) else FAIL
        c.detail = f"flash_attn_func 输出 {tuple(o.shape)}"

    @run("C3", "PyTorch SDPA（FlashAttention 的兜底路径）", False,
         "连 SDPA 都失败说明 torch 安装有更基础的问题，回到 B1/B2。")
    def _(c: Check):
        import torch
        import torch.nn.functional as F
        q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        c.status = PASS if bool(torch.isfinite(o).all()) else FAIL
        c.detail = f"SDPA 输出 {tuple(o.shape)}"

    @run("C4", "训练栈版本表", False, "缺哪个装哪个。版本钉死并记进 run manifest。")
    def _(c: Check):
        vers: dict[str, str] = {}
        for mod in ("transformers", "peft", "trl", "accelerate", "datasets",
                    "deepspeed", "liger_kernel"):
            try:
                m = __import__(mod)
                vers[mod] = str(getattr(m, "__version__", "?"))
            except Exception as e:
                vers[mod] = f"MISSING ({type(e).__name__})"
        c.data["versions"] = vers
        missing = [k for k, v in vers.items() if v.startswith("MISSING")]
        c.status = PASS if not missing else WARN
        c.detail = ", ".join(f"{k}={v}" for k, v in vers.items())


# ══════════════════════════════════════════════════════════════════════
# G · Qwen3.5 GDN 混合架构专项  ★ v2 新增
# ══════════════════════════════════════════════════════════════════════

def checks_gdn(with_model: bool) -> None:
    section("G · Qwen3.5 GDN 混合架构专项 ★ 静默回退是本项目最危险的坑")

    @run("G1", "causal_conv1d 真实 CUDA kernel 执行（不只是 import）", True,
         "★★★ 缺失时 Qwen3.5 会【静默回退】到慢且更吃显存的 PyTorch 实现——"
         "不报错不告警，你的吞吐/显存/MFU 全错而不自知。\n"
         "     解法：pip install causal-conv1d（可能需从源码编译，指定 sm_120）。\n"
         "     装不上 → Qwen3.5 线整体移到租的 A100（sm_80 栈成熟），"
         "本地 5090 只服务 Arctic-7B。这不影响 L0 必达层。")
    def _(c: Check):
        import torch
        from causal_conv1d import causal_conv1d_fn  # type: ignore
        b, d, seq, k = 2, 128, 64, 4
        x = torch.randn(b, d, seq, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(d, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        out = causal_conv1d_fn(x, w, bias, activation="silu")
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out).all())
        c.data.update(shape=list(out.shape), finite=ok)
        c.status = PASS if ok else FAIL
        c.detail = f"CUDA kernel 实际执行成功，输出 {tuple(out.shape)}"

    @run("G2", "fla (flash-linear-attention) 真实 kernel 执行", True,
         "★★★ 同 G1：缺失导致 GDN 静默回退。pip install flash-linear-attention。\n"
         "     装不上 → Qwen3.5 线移到 A100。")
    def _(c: Check):
        import torch
        import fla  # type: ignore
        fn = None
        for path, name in [("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
                           ("fla.ops.delta_rule", "chunk_delta_rule")]:
            try:
                fn = getattr(__import__(path, fromlist=[name]), name)
                c.data["kernel"] = f"{path}.{name}"
                break
            except Exception:
                continue
        c.data["fla_version"] = str(getattr(fla, "__version__", "?"))
        if fn is None:
            c.status = WARN
            c.detail = f"fla {c.data['fla_version']} 已装但未找到已知 GDN kernel 入口（API 可能改名）"
            return
        B, T, H, K, V = 1, 64, 4, 64, 64
        kw = dict(device="cuda", dtype=torch.bfloat16)
        out = fn(q=torch.randn(B, T, H, K, **kw), k=torch.randn(B, T, H, K, **kw),
                 v=torch.randn(B, T, H, V, **kw),
                 g=torch.rand(B, T, H, device="cuda", dtype=torch.float32).log(),
                 beta=torch.rand(B, T, H, **kw))
        o = out[0] if isinstance(out, (tuple, list)) else out
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(o).all())
        c.status = PASS if ok else FAIL
        c.detail = f"fla {c.data['fla_version']} GDN kernel 实际执行成功"

    @run("G3", "transformers ≥ 5.2.0 且能构造 Qwen3.5 配置", True,
         "Qwen3.5 需要 transformers>=5.2.0。pip install -U transformers。")
    def _(c: Check):
        import transformers
        v = transformers.__version__
        parts = []
        for p in v.split(".")[:3]:
            digits = "".join(ch for ch in p if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        ok_ver = tuple(parts) >= (5, 2, 0)
        c.data["transformers_version"] = v
        cfg_ok, cfg_err = False, ""
        try:
            from transformers import Qwen3_5TextConfig  # type: ignore
            cfg = Qwen3_5TextConfig()
            c.data["layer_types_sample"] = list(getattr(cfg, "layer_types", []))[:8]
            cfg_ok = True
        except Exception as e:
            cfg_err = f"{type(e).__name__}: {e}"
        c.data["config_class_ok"] = cfg_ok
        c.status = PASS if (ok_ver and cfg_ok) else FAIL
        c.detail = f"transformers {v} (>=5.2.0: {ok_ver}), Qwen3_5TextConfig: {cfg_ok}"
        if cfg_err:
            c.error = cfg_err

    @run("G4", "Qwen3.5-9B 配置的层型分布与 KV 成本核算", False,
         "取不到就用文档值（32层 = 24 GDN + 8 full attn；full: 4 KV头 × dim 256 "
         "→ 32 KiB/token）。下载模型后必须复核，KV 算错整个容量规划就错。")
    def _(c: Check):
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        lt = list(getattr(text_cfg, "layer_types", []) or [])
        n_full = sum(1 for t in lt if "full" in str(t).lower())
        n_lin = len(lt) - n_full
        kvh = getattr(text_cfg, "num_key_value_heads", None)
        hd = getattr(text_cfg, "head_dim", None)
        c.data.update(n_layers=len(lt), n_full_attention=n_full, n_linear=n_lin,
                      num_key_value_heads=kvh, head_dim=hd)
        if n_full and kvh and hd:
            kv_b = 2 * n_full * int(kvh) * int(hd) * 2
            c.data["kv_bytes_per_token"] = kv_b
            c.data["kv_kib_per_token"] = round(kv_b / 1024, 1)
            c.status = PASS
            c.detail = (f"{len(lt)}层 = {n_lin} GDN + {n_full} full；"
                        f"KV={kv_b/1024:.0f} KiB/token（文档值 32）")
        else:
            c.status = WARN
            c.detail = f"配置字段不全: layers={len(lt)} full={n_full} kvh={kvh} hd={hd}"

    if not with_model:
        print("  (未加 --with-model，跳过 G5/G6：需下载 ~18GB 权重)")
        return

    @run("G5", "vLLM 端到端加载 Qwen3.5-9B 并出 token", True,
         "★ 失败则 Qwen3.5 线整体移到 A100，本地 5090 只服务 Arctic-7B。"
         "L0 必达层不受影响。")
    def _(c: Check):
        from vllm import LLM, SamplingParams  # type: ignore
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            llm = LLM(model="Qwen/Qwen3.5-9B", max_model_len=4096,
                      gpu_memory_utilization=0.85, enforce_eager=False)
            outs = llm.generate(["SELECT 1;"], SamplingParams(max_tokens=16, temperature=0))
            msgs = [str(x.message) for x in w]
        # ★ 捕获回退警告——这是静默回退的第二道防线
        fb = [m for m in msgs
              if any(k in m.lower() for k in ("fall back", "fallback", "slow path",
                                              "not available", "causal_conv1d", "fla"))]
        c.data.update(n_outputs=len(outs), fallback_warnings=fb[:5])
        if fb:
            c.status = FAIL
            c.detail = f"出 token 成功但检测到 {len(fb)} 条回退警告 ← 走的是慢路径"
            c.error = fb[0][:300]
        else:
            c.status = PASS
            c.detail = "加载并生成成功，无回退警告"

    @run("G6", "文本模式下 vision tower 是否被跳过（否则白占显存）", False,
         "若 vision tower 被加载，用 Qwen3_5ForCausalLM + Qwen3_5TextConfig 显式走文本路径。")
    def _(c: Check):
        from transformers import AutoConfig  # type: ignore
        import torch
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
        has_vision = hasattr(cfg, "vision_config")
        alloc = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
        c.data.update(has_vision_config=has_vision, cuda_alloc_gb=alloc)
        c.status = WARN if has_vision else PASS
        c.detail = ("配置含 vision_config ← 确认服务时未加载 vision tower"
                    if has_vision else "无 vision 分支")


# ══════════════════════════════════════════════════════════════════════
# D · 推理侧
# ══════════════════════════════════════════════════════════════════════

def checks_serve() -> None:
    section("D · 推理侧（L0 必达层的地基）")

    @run("D1", "vLLM 导入与版本", True,
         "★ 推理线是 L0 必达层地基，这条 FAIL 整个项目停摆。"
         "装 vLLM ≥0.17（SM120 FP8 与 GDN 支持下限）。")
    def _(c: Check):
        import vllm  # type: ignore
        c.data["vllm_version"] = str(getattr(vllm, "__version__", "?"))
        c.status = PASS
        c.detail = f"vLLM {c.data['vllm_version']}"

    @run("D2", "vLLM 关键开关存在性（prefix caching / 推测解码 / 量化 / MTP）", False,
         "参数名跨版本改过。用 `vllm serve --help` 核对当前版本真实参数名，"
         "写进 configs/serve/。不要照抄旧博客。")
    def _(c: Check):
        if not shutil.which("vllm"):
            c.status = SKIP; c.detail = "vllm CLI 不在 PATH"; return
        rc, out, err = sh(["vllm", "serve", "--help"], timeout=180)
        text = (out + "\n" + err).lower()
        wanted = {"prefix_caching": "prefix-caching", "speculative": "speculative",
                  "quantization": "quantization", "gpu_mem_util": "gpu-memory-utilization",
                  "max_model_len": "max-model-len", "mtp": "mtp"}
        found = {k: (v in text) for k, v in wanted.items()}
        c.data["flags"] = found
        miss = [k for k, v in found.items() if not v]
        c.status = PASS if not miss else WARN
        c.detail = f"未匹配: {miss or '无'}（mtp 未匹配属正常，可能在 speculative 子配置里）"

    @run("D3", "GPU 指标：DCGM PROF 字段 vs nvidia-smi", False,
         "★ PROF 字段在 GeForce 上大概率不可得。兜底方案更好：用吞吐反算 MFU/MBU。"
         "简历上不要写 GPU_UTIL 百分比——它是时间占用标志不是工作量度量。")
    def _(c: Check):
        rc, out, _ = sh(["nvidia-smi",
                         "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                         "power.draw,temperature.gpu", "--format=csv,noheader"])
        c.data["nvidia_smi"] = out if rc == 0 else f"rc={rc}"
        prof_ok = None
        if shutil.which("dcgmi"):
            rc2, out2, err2 = sh(["dcgmi", "dmon", "-e", "1002,1004,1005", "-c", "1"], 30)
            prof_ok = rc2 == 0 and "N/A" not in out2
            c.data["dcgm_raw"] = (out2 or err2)[:400]
        c.data.update(dcgmi_present=bool(shutil.which("dcgmi")), dcgm_prof_available=prof_ok)
        if prof_ok:
            c.status = PASS; c.detail = "DCGM PROF 可用（SM_ACTIVE/TENSOR_ACTIVE/DRAM_ACTIVE）"
        elif rc == 0:
            c.status = WARN; c.detail = "PROF 不可用 → 改用 MFU/MBU 反算"
        else:
            c.status = FAIL; c.detail = "nvidia-smi 查询也失败"


# ══════════════════════════════════════════════════════════════════════
# E · 硬件实测
# ══════════════════════════════════════════════════════════════════════

def checks_hardware() -> None:
    section("E · 硬件实测（校准全部排程假设）")

    @run("E1", "实测 BF16 算力 TFLOPS（校准 SFT 耗时）", False,
         "测不出就沿用 209 TFLOPS 假设，但 manifest 里标为'未校准'。"
         "偏差 >30% 时 SFT 排程必须重算。")
    def _(c: Check):
        import torch
        n, iters = 8192, 20
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            _ = a @ a
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = a @ a
        torch.cuda.synchronize()
        tflops = (2 * n ** 3 * iters) / (time.time() - t0) / 1e12
        dev = abs(tflops - 209) / 209
        c.data.update(measured_bf16_tflops=round(tflops, 1), assumed_tflops=209,
                      deviation_pct=round(dev * 100, 1))
        c.status = PASS if dev < 0.30 else WARN
        c.detail = (f"实测 {tflops:.1f} TFLOPS（假设 209，偏差 {dev*100:.0f}%）"
                    + ("  ← 需重算排程" if dev >= 0.30 else ""))

    @run("E2", "实测显存带宽 GB/s（MBU 分母）", False,
         "测不出就用规格值 1792 GB/s，并在报告里注明用的是规格值。")
    def _(c: Check):
        import torch
        n, iters = 256 * 1024 * 1024, 20
        x = torch.empty(n, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            y.copy_(x)
        torch.cuda.synchronize()
        gbs = (2 * x.numel() * x.element_size() * iters) / (time.time() - t0) / 1e9
        c.data.update(measured_bw_gbs=round(gbs, 1), spec_bw_gbs=1792)
        c.status = PASS
        c.detail = f"实测 {gbs:.0f} GB/s（规格 1792，达成 {gbs/1792*100:.0f}%）copy 基准偏保守"


# ══════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════

def summarize() -> int:
    print(f"\n{'═' * 72}\n汇总\n{'═' * 72}")
    n = {s: len([c for c in RESULTS if c.status == s]) for s in (PASS, FAIL, WARN, SKIP)}
    print(f"\nPASS {n[PASS]} | FAIL {n[FAIL]} | WARN {n[WARN]} | SKIP {n[SKIP]}")

    crit = [c for c in RESULTS if c.critical and c.status == FAIL]
    other = [c for c in RESULTS if not c.critical and c.status in (FAIL, WARN)]

    def dump(title: str, items: list[Check], bar: str) -> None:
        if not items:
            return
        print(f"\n{bar * 72}\n{title}\n{bar * 72}")
        for c in items:
            print(f"\n  {'✗' if c.status == FAIL else '!'} [{c.id}] {c.name}")
            print(f"     现象: {c.detail}")
            if c.error:
                print(f"     错误: {c.error}")
            print(f"     兜底: {c.fallback}")

    dump("关键失败 —— 不解决不要开工", crit, "!")
    dump("非关键失败 —— 影响对应 JD 项，按兜底处理", other, "-")

    skipped = [c.id for c in RESULTS if c.status == SKIP]
    if skipped:
        print(f"\n跳过（依赖未装，装上后重跑）: {', '.join(skipped)}")

    print(f"\n{'═' * 72}\n计划决策建议\n{'═' * 72}")

    # 决策 1：Qwen3.5 线放本地还是 A100
    # ★ SKIP 绝不能当作健康。"没测"不是"通过"——这正是本项目要防的假阳性。
    gdn_required = ("G1", "G2", "G3")          # G5 需 --with-model，单列
    gdn = [c for c in RESULTS if c.id in gdn_required]
    gdn_pass = [c for c in gdn if c.status == PASS]
    gdn_bad = [c for c in gdn if c.status in (FAIL, WARN)]
    gdn_skip = [c for c in gdn if c.status == SKIP]
    g5 = next((c for c in RESULTS if c.id == "G5"), None)

    if gdn_bad:
        print(f"\n  → GDN 依赖不健康（{', '.join(c.id for c in gdn_bad)}）。")
        print("    Qwen3.5-9B 线整体移到租的 A100（sm_80 栈成熟），")
        print("    本地 5090 只服务 Arctic-7B。L0 必达层不受影响。")
        print("    ★ 绝不要在快 kernel 未生效的情况下跑压测——数字会全错且不报错。")
    elif gdn_skip:
        print(f"\n  → GDN 依赖未测（{', '.join(c.id for c in gdn_skip)} = SKIP，依赖未装）。")
        print("    ★ 未测 ≠ 通过。在 G1/G2/G3 全部 PASS 之前，")
        print("      不得启动任何 Qwen3.5 的压测或训练——静默回退不会报错。")
        print("    先装 causal-conv1d + flash-linear-attention + transformers>=5.2.0 再重跑。")
    elif len(gdn_pass) == len(gdn_required):
        if g5 is None or g5.status == SKIP:
            print("\n  → G1/G2/G3 已通过，但端到端未验证（G5 需 --with-model）。")
            print("    正式压测前请跑一次 `--with-model` 确认 vLLM 侧无回退警告。")
        elif g5.status == PASS:
            print("\n  → GDN 快 kernel 端到端验证通过，Qwen3.5-9B 可在本地服务。")
            print("    每个 run 的 manifest 仍要记录 G1/G2/G5 状态，未验证的 run 标 degraded。")
        else:
            print("\n  → G1/G2/G3 通过但 G5 端到端失败：vLLM 侧仍走回退路径。")
            print("    Qwen3.5 线移到 A100，或先解决 vLLM 的 GDN 后端问题。")

    # 决策 2：训练线放哪
    tr = [c for c in RESULTS if c.id in ("B4", "B5", "C1", "C2")]
    tr_bad = [c for c in tr if c.status in (FAIL, WARN, SKIP)]
    if len(tr_bad) >= 2:
        print(f"\n  → 训练侧依赖 ≥2 项不健康（{', '.join(c.id for c in tr_bad)}）。")
        print("    强烈建议训练侧整个租 A100/L40S。理由是确定性不是性能。")
        print("    附带收益：8B 全参可行、ZeRO/FSDP 能真跑、两条线并行不抢卡。")
    elif tr:
        print("\n  → 训练侧依赖健康。但多卡 ZeRO/FSDP 仍建议租卡")
        print("    （Blackwell 双卡 backward 有已记录的驱动崩溃）。")

    # 决策 3：用实测算力重算 SFT
    e1 = next((c for c in RESULTS if c.id == "E1"), None)
    if e1 and "measured_bf16_tflops" in e1.data:
        t = e1.data["measured_bf16_tflops"]
        print(f"\n  → 实测算力 {t} TFLOPS。SFT 重算（Qwen3.5-9B, LoRA, 2ep@2500tok）：")
        for label, ns in (("bird23-train-filtered 6601", 6601), ("全量 16.4k", 16400)):
            for coef, tag in ((4.5, "无梯度检查点"), (6.5, "开梯度检查点")):
                for mfu in (0.30, 0.40):
                    h = coef * 9.0e9 * ns * 2500 * 2 / (t * 1e12 * mfu) / 3600
                    print(f"     {label:28s} {tag} MFU{mfu:.0%} → {h:5.1f} h")

    # 决策 4：KV 核算
    g4 = next((c for c in RESULTS if c.id == "G4"), None)
    if g4 and "kv_kib_per_token" in g4.data:
        kv = g4.data["kv_kib_per_token"]
        print(f"\n  → Qwen3.5-9B 实测 KV = {kv} KiB/token（文档 32）。")
        if abs(kv - 32) > 4:
            print("    ★ 与文档值偏差较大，容量规划表必须用此值重算。")

    return 1 if crit else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过算力/带宽实测")
    ap.add_argument("--with-model", action="store_true", help="含 vLLM 实际加载 Qwen3.5-9B（~18GB）")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("═" * 72)
    print("ModelHub Day-0 预检 v2")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]} | 平台: {platform.platform()}")
    print("═" * 72)

    checks_system()
    checks_torch()
    checks_train()
    checks_gdn(with_model=args.with_model)
    checks_serve()
    if args.quick:
        print("\n  (--quick 已跳过 E 段硬件实测)")
    else:
        checks_hardware()

    code = summarize()

    if args.json:
        payload = {"timestamp": time.time(), "python": sys.version,
                   "platform": platform.platform(),
                   "args": {"quick": args.quick, "with_model": args.with_model},
                   "checks": [asdict(c) for c in RESULTS]}
        with open(args.json + ".tmp", "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(args.json + ".tmp", args.json)  # 原子写
        print(f"\n结果已写入 {args.json}")

    print(f"\n退出码 {code} ({'有关键失败' if code else '无关键失败'})")
    return code


if __name__ == "__main__":
    sys.exit(main())
