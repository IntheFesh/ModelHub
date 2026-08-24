# ModelHub · Interview Q&A Checklist

PLAN.md A12 item 5: every round's "面试追问" (interview follow-up
questions), verbatim from the original round prompts, as a checklist.
**Answers are intentionally left blank** — CLAUDE.md §7 (per PLAN.md's
own closing note: "把该轮「面试追问」自己答一遍...这是这个项目的真正
验收标准，代码过了 make verify 只是必要条件") makes answering these a
human step, not something this script/session should fill in.

Of the 19 rounds (Round 0, A0–A12, B1–B5), **12 carried an explicit
面试追问 section** in the original prompt text; the other 7 (Round 0,
A0, A1, A4, A6, A12, B3) did not — listed below with that stated
plainly, not backfilled with invented questions.

---

## A2 · SQL 执行沙箱

- [ ] 为什么 SQLite 的 timeout 拦不住笛卡尔积？
- [ ] DB 文件缺失为什么不能算模型答错？

## A3 · 结果比对器 ★ 全项目最重要

- [ ] 为什么不直接用官方脚本当 GRPO 奖励函数？
- [ ] "错判成对"和"对判成错"，哪个对训练更致命？为什么？
- [ ] 空对空算相等吗？大量空对空会怎样？

## A5 · 服务层 + Prompt/Schema 设计

- [ ] prefix caching 的命中条件？为什么 prompt 字段顺序影响命中率？
- [ ] GDN 的线性注意力层为什么没有随长度增长的 KV？它的前缀能复用吗？

## A7 · 监控（含 MFU/MBU）

- [ ] TTFT 和 TPOT 分别受什么影响？
- [ ] decode 阶段 MBU 高 MFU 低说明什么？

## A8 · 压测 + 容量规划 + 成本模型

- [ ] 单卡多少 QPS、P99、瓶颈在哪？
- [ ] 流量涨 10 倍先撑不住的是哪个环节（不一定是 GPU）？
- [ ] 两个模型的 KV 曲线为什么会交叉，交叉点由什么决定？

## A9 · 模型注册表 + 准入门禁

- [ ] 为什么净提升为正还要拦？
- [ ] 回归集怎么选，会不会过拟合到回归集？
- [ ] 门禁自己挂了 fail-open 还是 fail-closed，为什么？

## A10 · 灰度发布 + 自动回滚 + 线上抽样验证

- [ ] 5% 流量下多久能判断出 3% 的准确率劣化？
- [ ] 线上没有 gold SQL，你怎么知道准确率降了？

## A11 · 推理优化实验编排

- [ ] AWQ 与 GPTQ 原理差别？为什么量化后吞吐会涨？
- [ ] 推测解码为什么能加速（不是因为小模型快）？为什么高并发下收益归零？
- [ ] MTP 和 EAGLE 的区别？为什么 Arctic 用不了 EAGLE？

## B1 · 训练流水线 + 断点续训

- [ ] 断点续训除了权重还要存什么？
- [ ] 在混合架构上做 LoRA 有什么坑？你怎么发现的？

## B2 · 微调方式与分布式策略对比

- [ ] ZeRO-1/2/3 分别切什么？为什么 ZeRO-3 显存最省但可能更慢？
- [ ] FSDP 和 ZeRO-3 什么关系？
- [ ] QLoRA 的 4bit 量化的是什么，为什么不影响梯度？

## B4 · DPO

- [ ] DPO 和 SFT 的梯度有什么本质区别？
- [ ] 为什么这里可以不用 reward model？

## B5 · GRPO on veRL

- [ ] GRPO 和 PPO 的区别，为什么能去掉 critic？
- [ ] 组内 reward 全相同时会发生什么，你怎么检测？
- [ ] 你的奖励噪声从哪来，为什么系统错必须 mask 而不是给 0？

---

## Rounds with no 面试追问 in the original prompt

These rounds' original prompt text (verified against the source
CLAUDECODEPROMPTS.md upload, not reconstructed from memory) did not
include a 面试追问 section. Listed here rather than silently omitted,
so this checklist's coverage is auditable:

- Round 0 · 项目初始化
- A0 · 地基
- A1 · 数据层
- A4 · 评估器 + 报告
- A6 · 网关：鉴权/限流/配额/熔断 + 路由 + 计费
- A12 · 收尾 (this round)
- B3 · 坏模型生成器（门禁演练弹药）
