# ModelHub · 模型选型定稿

> 取代此前所有关于模型选择的说法。架构数据全部来自官方模型卡与文档，KV 计算式附上可复核。

---

## 一、定稿阵容

| 角色 | 模型 | 许可 | 为什么 |
|---|---|---|---|
| **生产主力** | `Snowflake/Arctic-Text2SQL-R1-7B` | Apache-2.0 | BIRD-dev 68.9% / Spider-test 88.8%，Text2SQL 专用；标准 GQA，KV 语义完全可预测 |
| **量化档** | 同上 AWQ / FP8 | 同上 | 同模型不同精度，路由与成本对照 |
| **自训线 + 架构对照** | `Qwen/Qwen3.5-9B` + 你的 LoRA | Apache-2.0 | 当代（Qwen3 已落后三代）；GDN 混合架构提供对照；**原生 MTP 头** |
| **兜底** | 任一商用 API | — | crossover 基准 |

**砍掉**：Qwen3-8B（落后三代，且 KV 成本是三者中最高的 2.57×）、
独立的通用 3B 小模型（通用小模型在 BIRD 上太差，路由过去会伤准确率，且多一个模型多一份维护）。

**Qwen3.8-27B 不用**（8/14 发布，27B BF16 约 54GB，单张 5090 装不下只能量化跑，
vLLM 成熟度未知）。但要写一条决策记录说明评估过。

---

## 二、架构核实与 KV 计算（可复核）

### Qwen3.5-9B（官方模型卡）

```
32 层 = 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
  → 24 个线性注意力层 + 8 个全注意力层

Gated Attention:  16 Q 头 / 4 KV 头 / head_dim 256 / RoPE dim 64
Gated DeltaNet:   32 V 头 / 16 QK 头 / head_dim 128
FFN intermediate: 12288（dense，非 MoE）
MTP:              trained with multi-steps  ← 原生多 token 预测头
Context:          262,144 原生
Vocab:            248,320（padded）
另含 vision encoder（是 VLM，文本任务用 Qwen3_5ForCausalLM + Qwen3_5TextConfig）
```

**KV 只在 8 个全注意力层上增长：**
```
KV/token = 2(K,V) × 8层 × 4头 × 256dim × 2byte = 32,768 B = 32 KiB
GDN 固定态 ≈ 12–24 MB / 序列（不随长度增长；开工时实测确认）
```

### 三模型对照

| 模型 | 层/KV头/dim | KV/token | 固定态 | 权重(BF16) |
|---|---|---|---|---|
| Qwen3-8B | 36 / 8 / 128 | 144 KiB | 0 | 16.4 GB |
| Arctic-7B | 28 / 4 / 128 | 56 KiB | 0 | 15.2 GB |
| **Qwen3.5-9B** | **8** / 4 / 256 | **32 KiB** | ~18 MB | 18.0 GB |

### 单卡 32GB 并发上限（KV预算 = 32×0.92 − 权重 − 2.5GB）

| prompt | Qwen3-8B | Arctic-7B | Qwen3.5-9B |
|---|---|---|---|
| 1k | 77 | **220** | 186 |
| 3k | 26 | **73** | **82** |
| 8k | 10 | 27 | **34** |
| 32k | 2 | 7 | **9** |

**★ 曲线在 1k–3k 之间交叉。** 短 prompt 时 Qwen3.5 的固定态是净负担，
长 prompt 时 GDN 的 O(1) 特性碾压。**这个交叉点就是你的路由判据**，
而且它是算出来再实测验证的，不是拍脑袋。

面试表述：
> "平台上两个模型的注意力架构不同，KV 经济学完全相反。Arctic 是标准 GQA，
> KV 随长度线性增长；Qwen3.5 是 3:1 的 GDN 混合，只有 1/4 的层有增长的 KV，
> 其余是固定大小的状态矩阵。所以短 prompt 时 Arctic 并发更高，
> 超过约 X token 之后 Qwen3.5 反超，到 8k 时领先 26%。
> 我的路由不是按'大模型小模型'分，是按 **schema 长度**分——
> 这个阈值是我压测出来的。"

---

## 三、四个陷阱（按危害排序）

### 陷阱 1 ★★★ · GDN 快 kernel 缺失会**静默回退**

HF 文档原文：GDN 路径（`Qwen3NextGatedDeltaNet`）需要 `causal_conv1d`（Dao-AILab）
和 `fla` 两个可选包，**没有它们模型会静默回退到更慢且更吃显存的 PyTorch 实现**。

**危害**：你的吞吐、显存、MFU/MBU 数字会全部错，而且不报错、不告警。
这正是 `CLAUDE.md` 反作弊条款针对的那类"看起来正常的假数据"。

**加重因素**：`causal_conv1d` 出自 Dao-AILab（FlashAttention 同一作者），
在 sm_120 上很可能有相同的编译问题。

**处置**：
- preflight 必须检查，且**不能只测能否 import**，要测**是否真的走了快 kernel**
  （对比开关两种路径的单步耗时，差异 <20% 说明快 kernel 没生效）
- 检测结果写进每个 run 的 manifest，未启用快 kernel 的 run 标 `degraded=true`，
  **不允许被任何压测报告引用**

### 陷阱 2 ★★ · 量化 + thinking 模式 = 输出截断

有实测报告：INT4 量化的 Qwen3.5 开 reasoning 后，约 70% 的 AIME25 答案因撞
32K 输出上限被截断（全精度约 30%）。原话是"权重几乎没掉点，它只是想太多没写完"。

**对 Text2SQL 是致命的**：SQL 写一半被截断，而截断的输出**绝不能算作模型答错**
（A4 里已经要求单列一类，现在有了具体的触发场景）。

**处置**：Text2SQL 场景**关闭 thinking 模式**。
但把"开 thinking + 量化"跑一次当作对照实验——截断率这个数字很值钱，
它是"量化影响的不只是权重精度"的实证。

### 陷阱 3 ★★ · LoRA 目标模块选错会只适配 25% 的层

默认的 `q_proj / k_proj / v_proj / o_proj` **只存在于 8 个全注意力层**，
24 个 GDN 层是完全不同的模块名。naive 的 LoRA 配置会只训到 1/4 的层。

**处置**：
- LoRA 目标必须包含 FFN（`gate_proj / up_proj / down_proj`，覆盖全部 32 层）
- 或显式列出 GDN 的模块名
- **训练启动前打印每层的可训练参数分布并断言覆盖全部 32 层**，
  这是 B1 的一条硬验收

这条本身是极好的面试素材：
> "在混合架构上做 LoRA 有个坑：默认 target_modules 只命中全注意力层。
> 我打印了逐层可训练参数分布才发现只覆盖了 8/32 层，加上 FFN 之后才是全覆盖。"

### 陷阱 4 ★ · 版本与依赖

- `transformers >= 5.2.0` 才支持 Qwen3.5
- vLLM ≥ 0.17 支持 GDN（有资料称在该版本上"生产稳定"，但需自行验证）
- Qwen3.5 是 VLM，文本任务用 `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig`，
  确认不会加载 vision tower（否则白占显存）

---

## 四、新增的三个实验（都是白捡的强素材）

### 实验 N1 · KV 经济学交叉点实测

按 prompt 长度扫描（1k/2k/3k/5k/8k/16k），两个模型各测最大并发与 SLA 内 QPS，
画出交叉曲线，找出真实交叉点。**路由阈值由此确定。**

预期结论：交叉点在 1k–3k 之间；实测值与上表估算的偏差本身要报告。

### 实验 N2 · MTP vs n-gram 推测解码对比 ★

这一项之前不可能做，现在可以了：
- **Qwen3.5-9B：原生 MTP 头**（模型自带，匹配的 draft 路径）
- **Arctic-7B：只能 n-gram**（无匹配的 EAGLE head，训一个是另一个项目）

两者各测加速比、接受率、随并发的收益衰减。

面试表述：
> "推测解码我测了两条路。Qwen3.5 有原生 MTP 头，接受率 X%；
> Arctic 没有匹配的 draft 模型，只能用 n-gram——但我的场景 SQL 大量逐字复制
> prompt 里的 schema 标识符，n-gram 接受率也有 Y%。
> 两者都在并发超过 Z 之后收益归零，因为 batch 已经填满，draft 计算反而挤占算力。
> 所以只在低峰期开，这是我的调度策略。"

**这个回答同时证明你懂 MTP、懂 EAGLE 为什么不可用、懂推测解码的加速原理
和它失效的边界。** 一个问题答完三层。

### 实验 N3 · GDN 的 MBU 对照

GDN decode 的算术强度低于标准注意力（<1 FLOP/B vs ~1），**比标准注意力更 memory-bound**。

两个模型同条件下测 MFU/MBU，做对照：
> "GDN 的 decode 比标准注意力更加 memory-bound，我实测两者的 MBU 差 X 个点。
> 这解释了为什么 Qwen3.5 虽然 KV 省，但单请求延迟并不更低——
> 它省的是显存容量，不是显存带宽。"

**"省的是容量不是带宽"这句话，是把两个瓶颈分清楚的证据。**

---

## 五、受影响的改动清单

### preflight.py 新增四项

```python
# G1 [CRITICAL] GDN 快 kernel 是否真的生效（不是能否 import）
#     测法：同一段 forward，装/不装 causal_conv1d+fla 两种路径的单步耗时
#     差异 <20% → 判定快 kernel 未生效 → FAIL
#     兜底：装 causal_conv1d + fla；装不上则 Qwen3.5 线整体移到 A100（sm_80 成熟）
# G2 transformers >= 5.2.0
# G3 vLLM 能否加载 Qwen3.5-9B 并正常出 token（GDN 路径端到端）
# G4 Qwen3_5ForCausalLM 文本模式下 vision tower 是否被跳过（显存占用验证）
```

### CLAUDE-CODE-PROMPTS.md 的改动

| 轮次 | 改什么 |
|---|---|
| **A1** | 无变化 |
| **A5** | 服务模型改为 Arctic-7B + Qwen3.5-9B 双模型；**新增：启动后从 vLLM 日志分别解析两者的 KV block 数与 GDN 状态占用写进 manifest**；新增 GDN 快 kernel 生效检测 |
| **A6** | 路由判据从"简单/复杂"改为**"schema token 长度阈值"**，阈值来自实验 N1 的实测交叉点 |
| **A8** | 压测扫描新增 **prompt 长度维度**（1k/2k/3k/5k/8k/16k），产出交叉曲线 |
| **A11** | 量化那组新增"thinking 开/关 × 量化"的截断率对照（陷阱 2）；推测解码那组改为 **MTP vs n-gram 双路对比**（实验 N2）；新增 MBU 对照（实验 N3） |
| **B1** | 基座 Qwen3-8B → **Qwen3.5-9B**；★ **LoRA target_modules 必须覆盖全部 32 层，启动前打印逐层可训练参数分布并断言**（陷阱 3）；关闭 thinking 模式 |
| **B2** | 六组对比不变，但要额外记录 GDN 层与全注意力层的显存/耗时占比 |
| **B5** | GRPO 基座同步换为 Qwen3.5-9B（或其小尺寸同族 2B/4B 求快） |

### 数字修订

此前所有基于 Qwen3-8B 的 KV/并发数字作废，改用 §二 的三模型表。

---

## 六、诚实边界

1. **我没有搜到比 Arctic-Text2SQL-R1 更新的 Text2SQL 专用开源模型**，
   我能查到的最新引用（2026 年 6 月论文）仍以 Arctic-32B 的 ~70% 为参照系。
   **但这不等于不存在。** 开工前自己看一遍 BIRD leaderboard（五分钟）。
   如果有更新的 7B 档专用模型，换掉主力即可——**平台代码一行不用改，
   这正好演示了多模型平台的价值**。

2. **GDN 固定态的 12–24 MB 是估算**，不同资料给的算法不一致。
   开工时从 vLLM 启动日志或显存实测确认，确认后 §二 的并发表要重算。

3. **Qwen3.5-9B 在 BIRD 上的表现我没有数据。** 它是通用 VLM，
   zero-shot 大概率明显低于 Arctic 的 68.9%。这不是问题——
   它在这个项目里的角色是**自训基座 + 架构对照**，不是准确率主力。
   你 LoRA 之后的分数与 Arctic 的差距，恰恰是"为什么生产默认路由走 Arctic"的依据。

4. **vLLM 对 GDN + sm_120 组合的支持成熟度**只能实测。
   若 preflight G3 失败，兜底是把 Qwen3.5 线整体移到租的 A100（sm_80 栈成熟），
   本地 5090 只服务 Arctic。这不影响 L0 必达层。
