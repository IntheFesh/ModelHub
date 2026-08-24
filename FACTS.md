# ModelHub · 事实登记与残余不确定性

> 合并此前的 FACTS-AND-RISKS 与 FINAL-AUDIT（已归档）。
> 模型架构部分见 `MODEL-SELECTION-FINAL.md`，本文件不重复。
>
> 分级：**A 级 = 已核实（有公开来源）** / **B 级 = 已推算（算式附上）** /
> **C 级 = 必测（preflight 覆盖）** / **D 级 = 未验证（你来确认）**
>
> **对外纪律：简历与面试只允许引用 A 级和你自己跑出来的数字。**
> B 级只用于排程；C 级必须先被 preflight 测掉；D 级必须先被你确认。

---

## 一、A 级 · 已核实

### 数据集

| 事实 | 值 | 对计划的影响 |
|---|---|---|
| BIRD 总规模 | 12,751 对 / 95 库 / **33.4 GB** | 下载必须最先启动 |
| BIRD train | 9,428 条 | — |
| BIRD train 中 **gold SQL 无法执行** | **425 条**（→ 可用 9,003） | 约 4.5%。验证了「gold 执行失败必须计数不能静默丢弃」这条规则，数字本身即素材 |
| BIRD dev | 1,534（简单 925 / 中等 465 / 挑战 144） | 难度分层现成 |
| Spider train / dev / test | 8,659 / 1,034 / 2,147 | — |
| Spider dev 难度 | 易 248 / 中 446 / 难 174 / 极难 166 | 分层评估现成 |
| BIRD 许可 | CC BY-SA 4.0 | 可用 |

### ★ 两个官方资产（都省时间）

**`birdsql/bird23-train-filtered`** —— 官方高质量过滤子集 **6,601 / 9,428（约 70%）**，
明确定位为 text-to-SQL 微调的 drop-in 替代。
→ 直接用它做 SFT。省掉自建执行验证过滤，训练时间从 13.1h 降到 5.3h。

**BIRD Mini-Dev V2** —— **780 条**（500 条从 dev 精选的 SELECT-only +
270 条新增，含 CRUD 与 JSON 操作），提供 **SQLite / MySQL / PostgreSQL 三方言**。
→ **这就是快评子集，而且是官方的**（比自抽 300 条更标准、可对外比较）。
→ 三方言直接坐实 JD 里「SQLite / PostgreSQL / DuckDB」那条，
且同一语义查询在不同方言下的执行差异是比对器的天然对抗样本。

**★ 270 条 CRUD 的特殊处置**：它们会和你的安全闸打架（安全闸拦 DELETE/UPDATE）。
- 快评集**只用 500 条 SELECT-only**
- **270 条 CRUD 转为安全闸的免费对抗测试集**：BI 场景下应 270/270 全部拦截，
  这就是安全闸检出率的实测数字——原本要自己造，现在官方送了

### 模型（详见 MODEL-SELECTION-FINAL.md）

| 事实 | 值 |
|---|---|
| Arctic-Text2SQL-R1-7B | **Apache-2.0**；底座 Qwen2.5-Coder-7B-Instruct；GRPO + 执行奖励训练 |
| Arctic-7B 成绩 | BIRD-dev **68.9%** / BIRD-test 68.5% / **Spider-test 88.8%** / Spider2.0-DK **15.6%** / EHRSQL 36.7% / ScienceBenchmark 51.8% / 平均 57.2% |
| Arctic-14B / 32B | BIRD 70.04% / 71.83% |
| Arctic 模型卡 out-of-scope | 明写「**无验证的生产系统**」，并要求验证生成的 SQL 以防数据泄露与越权 |
| Qwen3.5-9B | **Apache-2.0**；32 层 = 24 GDN + 8 全注意力；全注意力 4 KV 头 × dim 256；**原生 MTP 头**；262K 上下文；含 vision encoder |
| Qwen 版本线 | Qwen3.5(2026-02) → Qwen3.6 → Qwen3.7(闭源跳过) → Qwen3.8(2026-08)。**Qwen3-8B 落后三代** |

**★ Spider2.0-DK 只有 15.6%** —— 从 Spider-test 的 88.8% 掉到 15.6%。
这个断崖说明模型在学术基准上强，一碰企业级复杂 schema 就崩。
它**正好论证了你整套门禁与线上验证的必要性**：模型卡自己写着不适用于无验证的生产系统，
这个 15.6% 就是为什么。面试时这组对比比任何单个高分都有说服力。

### 硬件与框架

| 事实 | 值 |
|---|---|
| RTX 5090 | GB202 / sm_120 / 32GB GDDR7 / 1,792 GB/s / **无 NVLink** |
| vLLM sm_120 | v0.17+ 可用，有专门的 **SM120 FP8 GEMM** |
| FP8 在 5090 | **可用**（不需要 Hopper） |
| NVFP4 在 SM120 | 后端选择有 gap，部分路径回退 Marlin W4A16 |
| TensorRT-LLM 在 SM120/121 | trtllm-gen **FMHA cubin 缺失**；安装基本只能走 NGC 容器 |
| `DCGM_FI_DEV_GPU_UTIL` | **时间占用标志，非工作量度量**。1 个 SM 忙也读 100% |
| `DCGM_FI_PROF_*` | 文档标 "datacenter Volta 及以上" |
| Qwen3.5 GDN 依赖 | 需 `causal_conv1d` + `fla`，**缺失时静默回退**到慢且更吃显存的实现 |
| Qwen3.5 版本要求 | `transformers >= 5.2.0` |

### ★ sm_120 训练栈的六项已记录故障

全部有据可查，不是猜测：

| 故障 | 症状 | 根因 | 解法 |
|---|---|---|---|
| cuBLAS 缺 kernel | 第一次 matmul 就 `CUBLAS_STATUS_EXECUTION_FAILED` | torch cu128 无 sm_120 cuBLAS kernel | 升 torch 2.11+cu129 |
| 编译缓存污染 | 升级后同样报错 | 旧缓存持有不匹配 kernel | `rm -rf` 缓存目录 |
| **伪装成 OOM** ★ | `out of memory` 但**还剩 25GB+** | Triton/inductor kernel 在 sm_120 上启动失败 | `TORCHDYNAMO_DISABLE=1` |
| **PTX JIT 全废** ★ | `cpp_extension.load()` 编译失败 | **`libnvptxcompiler.so` 在 CUDA 12.8/12.9 安装包与官方镜像中缺失** | 补库 / 换镜像 / 用 AOT wheel |
| bitsandbytes 断裂 | `libnvJitLink.so.13`、`cdequantize_blockwise_fp32` 符号错 | bnb 编译于 cu12，torch 跳 cu130 | 钉 torch 在 cu129 |
| **多卡训练崩驱动** ★ | backward 阶段不可恢复的 driver crash | 双 Blackwell + gradient_checkpointing | 无稳定解 → 租卡 |

**打星三条分别威胁：整晚排错方向、Liger/FlashAttention 编译、ZeRO/FSDP 多卡对比。**

**结论：sm_120 推理栈成熟，训练栈是雷区。** 训练侧整体租 A100。

---

## 二、B 级 · 已推算（算式可复核）

### 三模型 KV 经济学

```
KV/token = 2(K,V) × 全注意力层数 × KV头数 × head_dim × 2 bytes
可用 KV 预算 = 32GB × 0.92 − 权重 − 2.5GB(运行时+激活)
```

| 模型 | 层/KV头/dim | KV/token | 固定态 | 权重 |
|---|---|---|---|---|
| Qwen3-8B（已弃用） | 36 / 8 / 128 | 144 KiB | 0 | 16.4 GB |
| Arctic-7B | 28 / 4 / 128 | **56 KiB** | 0 | 15.2 GB |
| Qwen3.5-9B | **8**（仅全注意力层） / 4 / 256 | **32 KiB** | ~18 MB | 18.0 GB |

### 单卡 32GB 并发上限

| prompt | Qwen3-8B | Arctic-7B | Qwen3.5-9B |
|---|---|---|---|
| 1k | 77 | **220** | 186 |
| 3k | 26 | 73 | **82** |
| 8k | 10 | 27 | **34** |
| 32k | 2 | 7 | **9** |

**★ 曲线在 1k–3k 之间交叉。** 短 prompt 时 Qwen3.5 的固定态是净负担，
长 prompt 时 GDN 的 O(1) 特性碾压。**这个交叉点就是路由阈值**，
由 A8 二维压测实测确定，实测与估算的偏差本身要写进报告。

### SFT 耗时（假设峰值 209 TFLOPS，LoRA fwd+bwd 系数 4.5×/6.5×）

| 数据集 | 5090 无梯度检查点 | 5090 开检查点 | A100 无 | A100 开 |
|---|---|---|---|---|
| bird23-filtered 6,601 × 2ep × 2500tok | 3.9–5.3 h | 5.7–7.6 h | 2.6–3.5 h | 3.8–5.1 h |
| 全量 16.4k | 9.8–13.1 h | 14–19 h | 6.5–8.7 h | 9.4–12.6 h |

**规则：50 步烟测出实测 tok/s 后，估算表作废，ETA 用实测重算。**
preflight E1 会实测峰值算力并自动重算这张表。

### 全参微调显存（AdamW）

| 模型 | 总计 | 32GB | 80GB |
|---|---|---|---|
| Qwen3.5-9B | ~134 GB | ✗ | ✗ 单卡；✓ 双卡 ZeRO-3 |
| Qwen3-8B | 119.2 GB | ✗ | ✓ 双卡 |
| Qwen3-1.7B | 25.3 GB | ✓ 勉强 | ✓ |

### 租卡成本（2×A100-80G，约 45 实例小时）

海外 $112–180；国内 ¥450–630。行情波动，以实际报价为准。

---

## 三、C 级 · 必测（preflight.py 覆盖）

| # | 待测 | 失败后果 | 兜底 |
|---|---|---|---|
| A1 | nvidia-smi / GPU 列表 | 全废 | 装 driver 570+ |
| A2 | 是否 WSL2 | CUDA graph 不可用，压测数字作废 | 迁原生 Linux |
| A3 | 磁盘 ≥150GB | 装不下 | 清盘 |
| A4 | `libnvptxcompiler.so` | PTX JIT 全废 | 换镜像 / 租卡 |
| B1–B3 | torch sm_120 / bf16 matmul / backward | 训练线全废 | torch 2.11+cu129 |
| B4 | torch.compile | 伪装成 OOM | `TORCHDYNAMO_DISABLE=1` |
| B5 | Triton kernel | Liger 死 | 租卡或降认知档 |
| C1 | bitsandbytes 4bit | QLoRA 死 | 钉 cu129 / 租卡 |
| C2 | FlashAttention | 训练加速那条空 | SDPA / FlashInfer / 租卡 |
| **G1** | **`causal_conv1d` 真实 kernel 执行** | **★★★ GDN 静默回退，所有数字错** | 装包 / Qwen3.5 线移 A100 |
| **G2** | **`fla` 真实 kernel 执行** | **★★★ 同上** | 同上 |
| G3 | `transformers ≥ 5.2.0` | Qwen3.5 加载不了 | 升级 |
| G4 | Qwen3.5 层型分布与 KV 核算 | 容量规划算错 | 用文档值并标注 |
| G5 | vLLM 端到端 + 无回退警告 | 走的是慢路径 | 移 A100 |
| G6 | vision tower 是否被跳过 | 白占显存 | 走文本类 |
| D1 | vLLM 导入 | **L0 地基，整个项目停摆** | 换版本 / FlashInfer |
| D2 | vLLM 参数名 | 实验跑不起来 | `vllm serve --help` 核对 |
| D3 | DCGM PROF 可得性 | GPU 利用率报不准 | 用 MFU/MBU 反算（更好） |
| E1 | **实测 BF16 TFLOPS** | 全部 ETA 未校准 | 沿用 209 但标注未校准 |
| E2 | 实测显存带宽 | MBU 算不出 | 用规格值并注明 |

---

## 四、GPU 利用率要换个报法

`DCGM_FI_DEV_GPU_UTIL`（= nvidia-smi 那个）的官方定义是
**「过去采样周期内至少有一个 kernel 在执行的时间占比」**——
时间占用标志，不是工作量度量。一个小 kernel 占住 1 个 SM 跑满采样窗口就读 100%，
其余 SM 全在睡。**而这正是 LLM 推理的常态。**

简历上写「GPU 利用率 95%」会被反问"你说的是哪个利用率"。

**优先方案**：`DCGM_FI_PROF_SM_ACTIVE` / `PIPE_TENSOR_ACTIVE`（算力瓶颈）/
`DRAM_ACTIVE`（带宽瓶颈）。**但 GeForce 上很可能不可得**，Day 0 测。

**兜底方案（其实更好）**：
```
MFU = (2 × 参数量 × 输出 tok/s) ÷ 峰值 BF16 FLOPS
MBU = (模型权重字节数 × 输出 tok/s) ÷ 峰值带宽
```
峰值用 preflight E1/E2 的**实测值**，不用规格值。

**对 LLM decode，MBU 高 + MFU 低 = memory-bound，这是正常且期望的状态。**
两个数硬件无关、口径清晰，**正好精确回答「瓶颈是显存、算力还是通信」**。

面试表述：
> "GPU_UTIL 不能用，它只是时间占用标志。我报的是 MFU 和 MBU：
> decode 阶段 MBU 达 X%、MFU 只有 Y%，说明是显存带宽瓶颈——
> 这也是为什么提高 batch size 能涨吞吐但不涨单请求速度。
> prefill 阶段反过来，MFU Z%，是算力瓶颈。
> 而且 GDN 的 decode 比标准注意力算术强度更低，我实测两个模型的 MBU 差 W 个点。"

---

## 五、D 级 · 残余不确定性（谁、何时消除）

| # | 事项 | 消除方式 | 时点 |
|---|---|---|---|
| 1 | GDN 固定态实际大小（12–24 MB 是估算） | vLLM 启动日志 / 显存实测 | D0 |
| 2 | Qwen3.5-9B 在 BIRD 上的分数 | 自己跑快评 | D0 夜 |
| 3 | Qwen3.5 层型分布与 KV/token 实际值 | preflight G4 + vLLM 日志 | D0 |
| 4 | vLLM 对 GDN + sm_120 的成熟度 | preflight G5 | D0 |
| 5 | 实测 TFLOPS / 带宽（校准全部 ETA） | preflight E1/E2 + 50 步烟测 | D0 |
| 6 | vLLM 当前版本的 MTP / 推测解码参数名 | `vllm serve --help` | D0 |
| 7 | 是否有比 Arctic 更新的 Text2SQL 专用模型 | 自己看 BIRD leaderboard（5 分钟） | 今天 |
| 8 | 租卡平台价格与现货 | 查平台报价 | 今天 |
| 9 | XGrammar 对 SQLite 子集语法的编译开销 | 实测 | D3 |
| 10 | veRL 当前版本对 Qwen3.5 的支持矩阵 | 查 release notes | D1 |

**7、8 今天就能免费消掉。**

---

## 六、我不知道的（诚实清单）

1. **我没搜到比 Arctic-Text2SQL-R1 更新的 Text2SQL 专用开源模型**，
   能查到的最新引用（2026 年 6 月论文）仍以 Arctic-32B 的 ~70% 为参照系。
   **但这不等于不存在。** 开工前自己看一遍 BIRD leaderboard。
   有更好的换掉主力即可——**平台代码一行不用改，这正好演示了多模型平台的价值**。

2. **Qwen3.5-9B 在 BIRD 上大概率明显低于 Arctic 的 68.9%**（它是通用 VLM）。
   这不是问题——它在项目里的角色是**自训基座 + 架构对照**，不是准确率主力。
   你 LoRA 后与 Arctic 的差距，恰恰是「生产默认路由走 Arctic」的依据。

3. **GDN 固定态与 vLLM 对混合架构 prefix caching 的实际复用比例**只能实测。
   若 preflight G1/G2/G5 失败，兜底是把 Qwen3.5 线整体移到 A100，
   本地 5090 只服务 Arctic。**L0 必达层不受影响。**

4. **我的容器无 GPU、无法访问你的环境**，本文件 C 级全部条目
   我一条都没有实测过。确定性只能来自你跑 `preflight.py`。
