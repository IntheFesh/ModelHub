# CLAUDE.md · ModelHub 工程宪法

> 放在仓库根目录，Claude Code 每个 session 自动读取。
> **本文件的约束高于任何单轮任务描述。** 冲突时停下来问，不要自作主张。

---

## 0. 项目是什么

多模型 LLM 推理服务平台，承载 Text2SQL 场景。平台上有自训模型走完
「数据 → 训练 → 评估 → 准入门禁 → 灰度上线 → 监控 → 迭代」全链路。

**判据是工业口径**：单卡多少 QPS、P99 多少、瓶颈在哪、怎么上线、怎么回滚、单请求多少钱。

**所有对外声称的数字必须能追溯到一次真实 run 的落盘产物。** 这是本文件存在的唯一理由。

---

## 1. 反作弊条款（最高优先级）

针对一类具体失败模式：为了让测试变绿、流程跑通，代码悄悄把真实行为换成假的。
**这类代码比不能运行的代码危害大一个数量级**——它污染训练奖励、污染评估数字、
污染面试时说出口的每一个百分比。

### 1.1 绝对禁止

```python
try:                                   # ❌ 吞异常返回默认值
    result = execute_sql(sql)
except Exception:
    return 0.0

try: ...                               # ❌ 裸 except / except-pass
except: pass

if os.environ.get("TESTING"):          # ❌ 测试环境分支
    return SAMPLE_RESULT

def compare_results(a, b):             # ❌ 无条件返回
    return True

def compute_p99(...):                  # ❌ 未实现却返回合法值
    return None   # TODO

rows = cursor.fetchall()[:1000]        # ❌ 静默截断
gpu_util = metrics.get("gpu_util", 0)  # ❌ 缺失指标补 0
```

### 1.2 强制写法

```python
def compute_p99(...):                                    # ✅ 未实现 → 硬失败
    raise NotImplementedError("compute_p99 pending A8; do not call")

try:                                                     # ✅ 分类 + 上下文 + 显式失败
    rows = execute_sql(sql, timeout_s=cfg.sql_timeout_s)
except SQLSyntaxError as e:
    return ExecOutcome.failure(ExecErrorCode.SYNTAX, sql=sql, db=db_id, cause=e)
except SQLTimeout as e:
    return ExecOutcome.failure(ExecErrorCode.TIMEOUT, sql=sql, db=db_id,
                               timeout_s=cfg.sql_timeout_s, cause=e)
# 未预期异常一律外抛，不捕获

@dataclass(frozen=True)                                  # ✅ 截断显式标记
class ResultSet:
    rows: list[tuple]
    truncated: bool
    row_limit: int
    rows_scanned: int | None    # 未知就是 None，不是 0

gpu_util: float | None = metrics.get("gpu_util")         # ✅ 没有就是没有
```

### 1.3 ★ 未测 ≠ 通过

**任何"跳过"「未安装」「无法验证」的检查项，一律不得被计为通过。**

这条适用于代码里的每一处健康判定、能力探测、依赖检测。典型反例：
一个依赖检测函数在 import 失败时返回 SKIP，而下游把「非 FAIL」当作「可用」。

判定逻辑必须是**白名单**（显式 PASS 才算通过），不能是黑名单（非 FAIL 就算通过）。

### 1.4 smoke test 的唯一合法形态

只允许**缩小输入规模**（20 条样本、1 个 db、64 max_tokens），
**不允许替换任何实现**。

- 禁止 `--fake-engine` / `MockComparator` 出现在 `src/` 下
- mock 只能在 `tests/`，且不得被 `src/` import
- CI 静态检查 `scripts/check_no_cheating.py`（AST 扫描，非正则）命中即 fail

### 1.5 元测试：证明测试能红

`tests/meta/` 必须有一组测试，作用是**注入必然失败的输入，断言对应检查会红**。
一个从来不会变红的测试等于没有测试。

至少覆盖：

| 元测试 | 注入 | 断言 |
|---|---|---|
| `test_gate_can_fail` | 准确率极低的假模型元数据 | 门禁判 REJECT 且失败码正确 |
| `test_comparator_can_say_unequal` | 确定不等的结果集对 | 返回 NOT_EQUAL |
| `test_regression_detector_fires` | 新模型在老模型答对的 case 上答错 | 回归检测触发 |
| `test_eval_fails_on_exec_error_flood` | 沙箱 80% 报错 | 拒绝出报告，而非给低分 |
| `test_report_rejects_missing_metric` | manifest 缺 p99 | 报告生成硬失败 |
| `test_skip_is_not_pass` | 依赖探测返回 SKIP | 下游判定为「不可用」而非「可用」 |

---

## 2. 错误模型

### 2.1 统一基类

```python
class ModelHubError(Exception):
    code: ErrorCode          # 枚举，穷举，不留 UNKNOWN 兜底
    stage: Stage             # DATA/EXEC/COMPARE/EVAL/TRAIN/SERVE/GATE/RELEASE
    context: dict[str, Any]  # 必含定位信息：run_id, db_id, sample_id, model_id
    retryable: bool          # 调用方据此决定重试，不许猜
    cause: Exception | None
```

### 2.2 错误码规则

允许一个 `UNCLASSIFIED`，但命中必须：WARN 日志 + 原始样本落盘到
`artifacts/runs/<run_id>/unclassified/` + 计数进 metrics。
**它是待办清单，不是垃圾桶。** 占比 > 1% → 评估/训练中止。

### 2.3 ★ SQL 执行错误六分类

全项目最重要的分类，直接决定 GRPO 奖励质量：

| 类别 | 含义 | 奖励语义 |
|---|---|---|
| `SYNTAX` | SQL 语法错 | 模型错，reward = 0 |
| `SEMANTIC` | 表/列不存在、类型不匹配 | 模型错，reward = 0 |
| `TIMEOUT` | 超时（可能是笛卡尔积） | 模型错，reward = 0，单独统计 |
| `EXEC_OK` | 跑通拿到结果集 | 交给比对器 |
| `HARNESS_DB_UNAVAILABLE` | 我们的 DB 挂了/文件缺失 | **系统错：mask 掉，不参与梯度** |
| `HARNESS_INTERNAL` | 我们代码的 bug | **系统错：mask + 告警** |

**把系统错当模型错是本项目最容易犯、最难发现、后果最严重的 bug。**
它让模型学噪声，而训练曲线看起来一切正常。

### 2.4 输出截断单列

模型输出被 `max_tokens` 截断导致 SQL 不完整，**绝不能算作「模型答错」**，
必须单列一类 `OUTPUT_TRUNCATED` 并统计占比。
（已知触发场景：量化模型 + thinking 模式会显著抬高截断率。）

---

## 3. 数字可追溯性

### 3.1 run manifest（强制）

任何产生数字的执行，必须写 `artifacts/runs/<run_id>/manifest.json`：

```json
{
  "run_id": "20260901-1432-a3f9c1",
  "git_sha": "…", "git_dirty": false,
  "config_hash": "sha256:…", "config_resolved": {},
  "dataset_version": "bird23-filtered@v1", "dataset_hash": "sha256:…",
  "eval_tier": "quick | full",
  "model_id": "qwen3.5-9b-sft-v4", "model_sha": "…", "adapter_sha": "…",
  "comparator_version": "2.1.0", "reward_fn_version": "1.3.0",
  "engine": {"name": "vllm", "version": "…", "commit": "…"},
  "kernel_status": {"causal_conv1d": true, "fla": true, "degraded": false},
  "hardware": {"gpu": "RTX 5090", "cc": "12.0", "count": 1,
               "driver": "…", "cuda": "…", "concurrent_procs": 1},
  "measured_peak_tflops": 198.4, "measured_bw_gbs": 1420.0,
  "seed": 42, "n_samples": 500,
  "error_breakdown": {"SYNTAX": 12, "TIMEOUT": 3, "OUTPUT_TRUNCATED": 5,
                      "HARNESS_INTERNAL": 0},
  "contaminated": false, "status": "COMPLETED"
}
```

`git_dirty: true`、`contaminated: true`、`degraded: true` 的 run
**一律不得被任何报告或简历引用**。CI 检查。

### 3.2 文档禁止手写数字

`docs/` 下所有性能与准确率数字由 `scripts/render_docs.py` 从 artifact 生成。
手写数字是简历事故的起点。

### 3.3 占位符

`[X] [TODO] [待测]` 必须机器可检：`scripts/check_placeholders.py`。
每轮收尾运行，投递前必须为 0。

### 3.4 缺失 ≠ 0

指标缺失就是 `None`。报告 schema 中 `required=True` 的字段为 `None` 时，
**报告生成器硬失败并列出缺失字段**，不得出一份看起来完整的假报告。

### 3.5 对外引用纪律

简历与面试**只允许引用**：(a) 有公开来源的第三方事实，(b) 你自己跑出来的、
manifest 完整且三个污染位均为 false 的数字。估算值只用于排程，不对外。

---

## 4. 配置

- Pydantic v2，全部 `ConfigDict(extra="forbid")`。**拼错的 key 必须报错**——
  静默忽略未知配置项是最经典的「训练结果对不上」来源。
- `lr / batch_size / seed / max_seq_len` 不给默认值，必须显式。
- 配置解析后计算 `config_hash` 写进 manifest。
- 改超参不改代码：超参全走 `configs/`，代码里出现魔法数字视为 bug。

---

## 5. 资源、外部调用与 GPU 纪律

### 5.1 外部调用

- **所有**网络请求 / 子进程 / DB 连接必须显式 timeout。无 timeout 视为 bug。
- 重试须：区分 retryable、有次数上限、指数退避+抖动、次数进 metrics。
  对 4xx 重试是 bug；对可重试错误不重试是脆弱。
- 资源用 context manager，异常路径也释放。
- 写文件**原子写**：`.tmp` → `os.replace()`。半截文件比没有文件更危险。

### 5.2 ★ GPU 独占规则

- 压测（bench）与训练（train）**禁止共享同一张 GPU**。共享条件下的
  QPS / P99 / 吞吐 / 利用率数字一律无效。
- 每个 bench run 启动前用 `nvidia-smi` 检查目标 GPU 有无其他进程，
  有则**拒绝启动**并报错。
- manifest 记录 `concurrent_procs`，> 1 的 run 标 `contaminated: true`。

### 5.3 ★ 环境指纹与内核状态

- vLLM / PyTorch / transformers / causal_conv1d / fla / veRL 版本
  + Docker 镜像 digest 全部写进 manifest。
- **GDN 快 kernel 状态必须显式记录**（`kernel_status`）。
  Qwen3.5 在 `causal_conv1d` / `fla` 缺失时会**静默回退**到慢且更吃显存的
  PyTorch 实现，不报错不告警。未验证快 kernel 的 run 标 `degraded: true`。
- 跨镜像 digest 的性能数字不可直接比较；报告生成器做横向对比时必须打警告。

---

## 6. 可中断与可恢复

### 6.1 ★ 过夜任务烟测前置（不可跳过）

任何预计 > 2 小时的无人值守任务，挂长跑前必须完成 20–50 步烟测并验证三项：

1. 显存峰值 < 可用显存 × 0.9
2. checkpoint 能原子落盘
3. `--resume-from` 恢复后 loss / metric 曲线与不中断一致

三项任一失败，**禁止挂长跑**。半小时成本换一个通宵。

### 6.2 ★ 夜间批任务编排

- 按风险从低到高排序，确定性任务（量化、评估）先跑
- 任务间 `;` 而非 `&&`，前序失败不阻断后续
- 每任务独立 manifest，`status ∈ {COMPLETED, FAILED, INTERRUPTED}`
- watchdog：任务死亡立即启动下一个，**禁止 GPU 空转**
- 队列**超配** 30–50%，跑不完自动顺延
- 队列快照落盘 `artifacts/queues/`

### 6.3 长跑任务硬要求

- 支持 `--resume-from`，且有测试断言续跑与不中断一致
- 中间结果每 N 步原子落盘，N 使单次丢失 ≤ 15 分钟工作量
- 捕获 `SIGTERM`/`SIGINT`：保存进度 → 干净退出
- 启动时打印预估耗时与显存峰值预测，**超出可用显存则拒绝启动**
  （OOM 要在第 0 步暴露，不是第 800 步）
- 无论成败都写 manifest
- **按时间预算跑，不按 epoch 跑**：设 `--max-hours`，到点取最佳 checkpoint

---

## 7. 数据与评估纪律

- train / dev / test 划分固定并做 hash 校验，CI 断言三者交集为空
- 训练期执行验证用的 DB 与评估用的 DB **物理隔离**（不同目录、只读挂载）
- 评估默认 greedy（`temperature=0`），采样类评估必须记录 seed 与 n
- 数据处理每步记录：输入 hash → 处理器版本 → 输出 hash

### 7.1 ★ 评估分档

- **快评**：官方 BIRD Mini-Dev V2 的 500 条 SELECT-only 固定子集，
  用于所有迭代对比与门禁初筛
- **全评**：完整 dev，仅用于基线、最终结果、门禁终判
- 快评子集 ID 列表固定并写进 manifest，跨实验不可变更
- **报告必须标注每个数字来自哪档，混用即为无效报告**
- gold SQL 执行结果按 `(db_id, gold_sql_hash, db_file_hash)` 缓存

---

## 8. 日志

- 结构化 JSON，固定字段 `run_id / stage / component / level`
- 异常日志必须带完整 traceback，禁止只打 `str(e)`
- 敏感信息脱敏（API key、DB 密码、token）
- **日志可以截断**（加 `…(truncated N chars)` 标记），**数据不可以**。
  两者代码路径必须分开。

---

## 9. 工程风格

- Python 3.11+，`ruff` + `mypy --strict`（`src/` 必须过）
- 类型注解完整，`Any` 需注释说明原因
- 函数超 50 行或圈复杂度超 10 → 拆
- 依赖锁定，关键库版本钉死并记进 manifest
- 明确的模块导出面，禁止跨层反向 import

---

## 10. 每轮任务的交付契约

Claude Code 完成任何一轮，必须同时给出：

1. **代码**：过 `ruff` + `mypy --strict` + 单元测试
2. **测试**：单元测试 + 至少一条元测试（证明这块检查能红）
3. **一条可执行验收命令**：`make verify-<id>`，打印通过/失败明细
4. **产物**：产生数字则 `artifacts/runs/<run_id>/` 有完整 manifest
5. **决策记录**：往 `docs/design-decisions.md` 追加一条（格式见 §11）
6. **诚实清单**：本轮**没做**什么、**假设**了什么、**已知局限**是什么

**任何一项缺失，这轮不算完成。**

---

## 11. 决策记录格式

```markdown
## DD-0012 · 推理引擎选型：vLLM over SGLang
- 日期：2026-09-05 · Run: 20260905-1102-7c2e11
- 决策：生产服务用 vLLM 0.x.y
- 考虑过：SGLang（吞吐更高）、TensorRT-LLM（SM120 上 FMHA cubin 缺失）
- 为什么选：同卡同 prompt 分布下 vLLM 吞吐 X vs SGLang Y，差距 Z%，
  但 vLLM 的 prefix caching 在固定 schema 前缀场景命中率 W%，实际收益反超
- 什么情况会失效：prompt 前缀不再固定 / 模型不再频繁迭代 / 需极致 TTFT
- 实测数字：artifacts/runs/20260905-1102-7c2e11/report.md
```

---

## 12. 交流方式

- 有歧义**先问再写**，不要挑一种解释闷头实现
- 发现任务描述里的错误、不可行处、更好做法 → 直接说，不要顺着做
- 不确定就说不确定。**「我不知道这个数字」永远好过一个编出来的数字**
- 不 claim novelty。文档中主动写清站在谁的肩上
  （vLLM / veRL / LLaMA-Factory / Spider / BIRD / Arctic-Text2SQL-R1 / Qwen）
