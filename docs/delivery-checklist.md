# ModelHub · A12 Delivery Checklist

PLAN.md A12's closing ask, answered item by item against this
session's actual state (no GPU, no real bench/eval/training run ever
executed in this sandbox — see docs/design-decisions.md throughout).
Regenerate the counts below with `make check-pollution` / by re-reading
`docs/design-decisions.md` and `docs/incident-log.md` directly; they are
not themselves script-rendered (unlike `docs/metrics-summary.md`),
since this checklist is a one-time human-facing summary, not a
recurring artifact.

## 六个核心数字是否都有 artifact

| # | 数字 | 状态 | 说明 |
|---|---|---|---|
| 1 | QPS | ✗ | 无真实 bench run；`docs/metrics-summary.md` 显式标 PENDING |
| 2 | P99 latency | ✗ | 同上 |
| 3 | MFU/MBU | ✗ | 同上 — `monitor/mfu_mbu.py`/`bench/experiments/mfu_mbu_comparison.py` 的公式代码已实现且有单元测试，只是没有真实测量输入 |
| 4 | 瓶颈分类 (compute/memory/balanced) | ✗ | 依赖 #3，同样 PENDING |
| 5 | 单请求成本 | ✗ | `gateway/billing.py`/`bench/cost_model.py` 的计算代码是真的，`configs/gateway/billing.yaml` 的单价是标注过的占位值（"non a market price"），不是真实 GPU-hour 成本模型算出来的数字 |
| 6 | Arctic/Qwen3.5 concurrency crossover 点 | △ | 见下一节 — 有真实计算，但输入是估算值，不是测量值 |

**六个数字，零个有真实测量 artifact。** 每一个背后的计算代码（capacity
planning、MFU/MBU、cost model、load test harness）都真实存在、有单元
测试、且已通过 `make verify-a8`/`make verify-a11`/`make verify-a12`——
缺的只是一台真实 GPU 去产生输入。这不是"没做"，是"代码做完了、等
真机跑"（用户原话："写出全部代码即可，后续GPU上面报错我会让你继续
修改的"）。

## ★ 两模型 KV 交叉曲线是否实测得出，路由阈值是否已回填

**否，两者都还是估算值，尚未实测。**

- `docs/crossover-curve.md` 由 `scripts/render_docs.py` 真实计算生成
  （复用 A8 `find_concurrency_crossover_point`，不是手写数字），在
  `weight_bytes`/`gdn_fixed_state_bytes` 均为文档估算值的前提下，算出
  交叉点在 **2000 prompt tokens**（Arctic-7B 97 并发 vs Qwen3.5-9B 103
  并发）——落在 MODEL-SELECTION-FINAL.md 文档记载的 1k–3k 估算区间内，
  内部一致，但仍然是"估算 → 估算"的推导，不是"实测 → 实测"。
- `configs/gateway/routing.yaml` 的 `schema_token_threshold: 1500` /
  `threshold_source: "estimate"` 未回填——保持 A6 round 写下时的原值，
  因为没有 A8 真实 prompt-length 压测扫描可以拿来回填。

## 门禁拦截与灰度回滚各几次，是否每条都有根因

| 类型 | 次数 | 根因是否记录 |
|---|---|---|
| GATE_REJECTION | 2 | 是 — `docs/incident-log.md` 两条均来自 `make demo` 真实跑出的 REJECT verdict，`reason` 字段写明具体是哪道闸（accuracy）、差多少（execution_accuracy 0.00% < required 50.00%） |
| CANARY_ROLLBACK | 0 | — 本沙箱从未运行过真实灰度流量，`release/rollback.py::trigger_rollback` 的调用点（`release/orchestrator.py::run_canary_cycle`）只在 A10 的单元/元测试里被触发过 |

`release/incident_log.py`（本轮新增，A9/A10 round 遗留的真实缺口——
"拦截记录 append-only 写 docs/incident-log.md" 此前从未真正接线）是
这两条记录能存在的原因；具体数字用
`python -c "from modelhub.release.incident_log import count_incidents; print(count_incidents())"`
可随时复现，不是手数的。

## 决策记录条数

**28 条**（`docs/design-decisions.md`，`DD-0001` 到 `DD-0028`），可用
`grep -c "^## DD-" docs/design-decisions.md` 复现。每条都含"考虑过的
替代方案 / 为什么选 / 什么情况会失效 / 实测数字"四段，格式见
CLAUDE.md §11。

## 哪些数字来自快评、哪些来自全评

**都不适用——本沙箱没有跑过任何一次真实的快评或全评。**
`eval/runner.py`/`eval/report.py` 的两档评估分离逻辑（CLAUDE.md
§7.1）已实现并有单元测试（`tests/unit/a4/`），`gate/admission.py`
五道闸的真实执行路径也在 `make demo` 里被证明可以跑通并给出正确
verdict——但"快评/全评"这个区分只在真的对着一个served模型跑
一遍评估时才有意义，本沙箱从未做到这一步。

## 还有哪些是估算未实测的（诚实清单）

- `configs/serve/model_profiles/{arctic_7b,qwen3_5_9b}.yaml` 的
  `weight_bytes`/`gdn_fixed_state_bytes` —— FACTS.md 文档估算值，
  待真实 vLLM 启动日志（`serve/vllm_log_parser.py`）回填
- `docs/capacity-plan.md`/`docs/crossover-curve.md` 的全部并发数字 ——
  上面两个估算值的直接推论，公式代码真实，输入不真实
- `configs/gateway/routing.yaml` 的 `schema_token_threshold` —— 见上，
  `threshold_source: estimate` 未变
- `configs/gateway/billing.yaml`/`configs/bench/experiments/mfu_mbu_
  comparison.yaml` 的价格与峰值 TFLOPS/带宽 —— 均标注"占位/未校准"，
  后者甚至直接写着 FACTS.md 的 spec-sheet 数字（"UNCALIBRATED"）
- A11 六组推理优化实验（量化/前缀缓存/推测解码/约束解码/引擎对比/
  MFU-MBU）—— 全部是真实编排代码 + 单元/元/烟测通过，零组产出过真实
  benchmark 数字（本沙箱无 GPU/无真实 vLLM·SGLang 服务实例）
- B1–B5 训练线的全部产出（SFT/DPO/GRPO 曲线、断点续训验证、微调方式
  对比表）—— 见下方任务列表，这些轮次在本 session 完成 A12 时尚未
  开始实现

## 已知局限（超出上面六项之外）

- `make demo` 用的是 `scripts/demo_stub_model_server.py`——一个显式
  标注"NOT A REAL MODEL"的固定响应桩，不是真实推理；它证明的是
  gateway/sqlexec/compare/gate/incident-log 这几条链路能端到端接起来，
  不证明任何准确率或性能数字
- `gateway/quota.py` 的配额强制发生在生成之后（因为这个模块唯一的
  入口 `enforce_quota` 需要用本次请求自己的 token 数才能判断），这
  意味着一次超额请求仍然会先花掉真实生成算力、再在返回前被拒绝——
  见 `gateway/app.py` 模块文档字符串与相关 DD 条目
