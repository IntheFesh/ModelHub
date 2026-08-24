# ModelHub

多模型 LLM 推理服务平台，承载 Text2SQL 场景。自训模型走完
「数据 → 训练 → 评估 → 准入门禁 → 灰度上线 → 监控 → 迭代」全链路。

判据是工业口径：单卡多少 QPS、P99 多少、瓶颈在哪、怎么上线、怎么回滚、单请求多少钱。
所有对外声称的数字必须能追溯到一次真实 run 的落盘产物 —— 见 `CLAUDE.md`。

## 文档

- [`CLAUDE.md`](CLAUDE.md) —— 工程宪法，约束高于任何单轮任务描述
- [`PLAN.md`](PLAN.md) —— 执行计划与排程
- [`FACTS.md`](FACTS.md) —— 已核实事实与残余不确定性分级登记
- [`MODEL-SELECTION-FINAL.md`](MODEL-SELECTION-FINAL.md) —— 模型选型定稿
- [`preflight.py`](preflight.py) —— Day-0 环境预检
- [`docs/design-decisions.md`](docs/design-decisions.md) —— 决策记录
- [`docs/build-log.md`](docs/build-log.md) —— 每日 build log

## 站在谁的肩上

vLLM / veRL / LLaMA-Factory / Spider / BIRD / Arctic-Text2SQL-R1 / Qwen。
本项目不 claim novelty，模型层直接用开源 SOTA；工程价值在网关、门禁、灰度、
容量规划、成本模型这些开源不提供的部分。

## 快速开始

```bash
make install       # 建虚拟环境 + 装依赖
make lint           # ruff
make typecheck      # mypy --strict
make test           # pytest（跳过需要真实 GPU / 外网的用例）
make verify-a0       # 单轮验收，例如 A0 地基层
```

## 目录结构

```
configs/{data,train,serve,gate,bench}/   配置（超参不进代码，见 CLAUDE.md §4）
src/modelhub/{common,data,sqlexec,compare,eval,train,
              serve,gateway,gate,release,bench,monitor}/
tests/{unit,golden,smoke,meta}/          meta/ 证明检查能红，见 CLAUDE.md §1.5
scripts/                                  check_no_cheating.py / check_placeholders.py / night_queue.py
docs/                                     design-decisions.md / interview-qa.md / architecture.md（数字由脚本生成，不手写）
artifacts/                                run 产物，gitignored；数字的唯一来源
```

## 已知环境局限（诚实清单）

本仓库的多数代码轮次是在无 GPU、无法访问 HuggingFace Hub 的沙箱环境中编写的。
CPU-only 且不依赖外网的部分（`common/`、`sqlexec/` 用 SQLite/DuckDB/PostgreSQL、
`compare/`、`gateway/` 用真实 Redis、`monitor/` 的 MFU/MBU 计算与 Prometheus 客户端）
已用真实依赖跑通并有单元测试。凡依赖真实 GPU（vLLM 加载模型、训练、压测出数字）
或依赖下载 BIRD/Spider/HF 模型权重的部分，代码与接口已写完，但标注为待真机验证
（`pytest.mark.requires_gpu` / `requires_network`，或 manifest 中显式字段），
绝不伪造通过。逐轮的诚实清单见对应的 `docs/design-decisions.md` 条目。
