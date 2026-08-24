# ModelHub · 决策记录

格式见 `CLAUDE.md` §11。按时间顺序追加，不改写历史条目。

---

## DD-0001 · Round 0 工具链选型

- 日期：2026-08-24 · Run: 无（本轮不产生数字，纯脚手架）
- 决策：`src/` layout + hatchling 构建、`ruff`（lint+format 一体）、
  `mypy --strict`（仅 `src/modelhub/`）、`pytest`（`tests/{unit,golden,smoke,meta}/`）、
  `pre-commit`。依赖用 `pyproject.toml` optional-dependencies 分组
  （`data/serve/gateway/monitor/train/dev`），避免每轮都装全部依赖。
- 考虑过：`black+flake8+isort` 三件套 —— 放弃，`ruff` 一个工具覆盖三者且快一个数量级；
  `poetry` —— 放弃，`hatchling` 更轻，不需要 poetry 的依赖解析器。
- 为什么选：单一工具链减少配置漂移；`mypy --strict` 只卡 `src/modelhub/`
  是因为 `tests/` 里大量用 `pytest.raises`/fixture 动态特性，strict 模式对测试代码
  收益低但摩擦高。
- 什么情况会失效：如果测试代码本身开始出现类型相关的 bug 频率上升，
  应该重新评估把 `tests/` 也纳入 `mypy --strict`。
- 实测数字：无（`make lint` / `make typecheck` 在空骨架上通过，无性能数字产出）。

---

## DD-0002 · A0 错误分类：单一扁平 ErrorCode，而非按 stage 拆分多个枚举

- 日期：2026-08-24 · Run: 无（本轮不产生数字，`make verify-a0` 61/61 测试通过是产物）
- 决策：`common/errors.py` 用**一个**扁平 `ErrorCode(StrEnum)` 覆盖全项目，
  而不是每个 stage/模块各自一个枚举。SQL 执行的七分类
  （SYNTAX/SEMANTIC/TIMEOUT/EXEC_OK/OUTPUT_TRUNCATED/HARNESS_DB_UNAVAILABLE/
  HARNESS_INTERNAL）也是这个枚举的成员；`ExecErrorCode = ErrorCode`
  只是一个别名，让 `sqlexec/` 里的调用点能照抄 CLAUDE.md §1.2 给出的写法
  （`ExecOutcome.failure(ExecErrorCode.SYNTAX, ...)`）。
- 考虑过：(a) 每个 stage 一个独立枚举（`ExecErrorCode`、`GateErrorCode`、
  `GatewayErrorCode`…），(b) 用字符串而非枚举（不做穷举校验）。
- 为什么选：manifest 的 `error_breakdown` 字段是一个 `{code: count}` 的扁平字典，
  横跨 EXEC/EVAL/GATE 等多个 stage 统计；如果每个 stage 一个枚举，
  这个字典的 key 集合就要在多个类型间做字符串级别的隐式统一，
  等于把"到底是不是同一套词表"这件事又用回了字符串约定——不如从一开始
  就是同一套词表。Python 的 `Enum` 一旦定义了成员就不能被子类化，
  所以选择"扁平但允许后续轮次继续往同一个类里加成员"，而不是
  "分散但各自独立演进"。
  另外把 SQL 六/七分类拆成两个集合常量 `SQL_MODEL_ERROR_CODES` /
  `SQL_SYSTEM_ERROR_CODES`（而不是让每个调用点各自判断"这个 code 算不算模型错"），
  是因为 B5（GRPO reward masking）和 A4（eval 报告的 HARNESS_* 占比熔断）
  都要做完全一致的这个判断——判断逻辑本身也要有唯一真源。
- 什么情况会失效：如果不同 stage 之间出现语义冲突的同名错误码需求
  （目前没有遇到），需要拆分或加前缀。
- 实测数字：无。`tests/meta/a0/test_check_no_cheating.py` 与
  `test_skip_is_not_pass.py` 证明反作弊静态扫描与"未测≠通过"白名单门可以真正变红，
  不是恒为绿的假测试。

---

## DD-0003 · 沙箱环境无 GPU/无 HuggingFace 访问：分层验证策略

- 日期：2026-08-24 · Run: 无
- 决策：本项目在无 GPU、无法访问 HuggingFace Hub、磁盘仅 30GB 的沙箱容器中开发。
  PyPI 与 apt 主源可用，因此：
  1. 纯 CPU 且不依赖外网的层（`common/`、`sqlexec` 用 SQLite/DuckDB/PostgreSQL、
     `compare/`、`gateway` 用真实 Redis、`monitor` 的 MFU/MBU 计算与
     Prometheus 客户端）用真实依赖跑通并有单元测试，不是"假装通过"。
  2. 依赖真实 GPU（vLLM 加载模型、实际训练、压测出数字）或依赖下载
     BIRD/Spider/HF 模型权重的部分，代码与接口写完，但用
     `pytest.mark.requires_gpu` / `requires_network` 标记为待真机验证，
     绝不用 mock 替换真实实现后谎称"跑通"（违反 CLAUDE.md §1.4 会是本项目
     最大的反讽）。
  3. 这些标记本身就是本轮及后续每轮"诚实清单"的机器可检来源——
     `pytest -m requires_gpu --collect-only` 能随时列出"还差什么真机验证"。
- 考虑过：伪造 GPU 检测结果或用极小 mock 模型冒充完整链路跑通——直接违反
  `CLAUDE.md` 反作弊条款，排除。
- 为什么选：CLAUDE.md 的核心诉求是"数字可追溯到真实 run"，而不是"每轮都必须有数字"。
  没有 GPU 就诚实地不产出 GPU 数字，比伪造更符合项目的第一性原则。
- 什么情况会失效：一旦有真实 GPU 机器可用，本决策的"待验证"标记必须逐条清空，
  而不是长期保留——`pytest.mark.requires_gpu` 是一个进度追踪机制，不是免死金牌。
- 实测数字：无。
