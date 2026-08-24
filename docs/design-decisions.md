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

---

## DD-0004 · 轮次执行顺序：A2（sqlexec）先于 A1（data）实际编码

- 日期：2026-08-24 · Run: 无
- 决策：`CLAUDE-CODE-PROMPTS.md` 里 A1（数据层）排在 A2（SQL 沙箱）之前，
  但 A1 的"gold SQL 执行验证"（已知 BIRD-train 425 条 gold 跑不通，
  必须落盘 `gold_exec_failures.jsonl`，不能静默丢弃）本身就需要执行 SQL。
  与其在 `data/` 里重新造一个简化版执行器（和 A2 要造的沙箱在只读连接、
  超时、错误分类上几乎是同一件事，写两遍才是真正的重复），
  直接让 `data/` 依赖 `sqlexec/` 的公开 API。
- 考虑过：(a) 在 A1 里写一个仅供 gold 校验用的迷你 sqlite3 执行器，
  不做进程隔离；(b) 照原顺序先写 A1 用 mock 占位再回填。
- 为什么选：(a) 会制造两份错误分类逻辑，两份代码迟早会在语义上飘走；
  (b) 直接违反 CLAUDE.md §1.4（mock 只能在 tests/，不得被 src/ import）。
  由于是在一个连续 session 里编写整个代码库（不是真的分成 19 个互不知情的
  session），提前调整内部实现顺序、保持轮次编号（A1/A2）不变，
  是 CLAUDE.md §12 "发现任务描述有错或有更好做法，直接说，不要顺着做"
  这条纪律要求的做法。`sqlexec` 不依赖 `data`（它是一个不知道 BIRD/Spider
  schema 的通用执行原语），所以 `data → sqlexec` 是正向依赖，不违反
  CLAUDE.md §9 的"禁止跨层反向 import"。
- 什么情况会失效：如果未来真的需要严格按 19 个独立 session 复现本项目
  （比如给另一个人逐轮验收），文档顺序仍按 A1→A2 编号，但代码交付顺序
  应参照本决策记录，不必强行按文档顺序逐字重现。
- 实测数字：无。

---

## DD-0005 · sqlexec 用"每任务一个子进程"而不是常驻 worker 池

- 日期：2026-08-24 · Run: 无（`make verify-a2` 37/37 测试通过是产物）
- 决策：`execute_isolated()` 每次调用都 `multiprocessing.Process(...).start()`
  一个全新子进程，`join(timeout_s)` 后若还活着就 `kill()`；并发靠
  `pool.execute_many()` 用 `ThreadPoolExecutor` 限制同时存在的子进程数，
  而不是维护一个常驻、循环从队列取任务的 worker 池。
- 考虑过：`concurrent.futures.ProcessPoolExecutor` + `future.result(timeout=)`。
  实测/查证：`ProcessPoolExecutor` 拿不到"单个任务超时就杀掉那一个 worker"
  的能力——`future.result(timeout=)` 超时只是不再等待，卡住的子进程
  （比如正在执行笛卡尔积的 SQLite C 扩展）仍在后台占着那个 worker 槽位
  继续跑，直到整个进程池关闭。这正是 A2 面试追问要考的点：
  "为什么 SQLite 的 timeout 参数拦不住笛卡尔积"——如果沙箱本身也只是
  "客户端不等了"而不是"真的把它杀掉"，就是同一个坑换了个地方犯。
- 为什么选：每任务一个进程虽然有 fork/spawn 开销（用 `spawn` 而非 `fork`，
  避免继承父进程的锁/线程状态导致死锁），但换来了任意一个任务都可以在
  不影响其它任务的前提下被真正 SIGKILL。`tests/unit/a2/test_pool.py::
  test_one_slow_task_does_not_block_the_rest_of_the_batch` 就是在验证这一点：
  一个必然跑 20 亿步的递归 CTE 和 5 个正常查询混在同一批里，
  批次总耗时被 timeout 严格限制住，而不是被那一个任务拖垮。
- 什么情况会失效：如果未来 QPS 需求高到"每任务一个进程"的 fork 开销
  本身成为瓶颈（BIRD/Spider 规模的评估目前测下来不会），
  需要换成常驻 worker 池 + 主进程主动 kill 对应 worker 再重启的模型——
  代价是要自己重写"kill 单个 worker 不影响其它 worker"这部分，
  ProcessPoolExecutor 不提供。
- 实测数字：无（性能数字要等真实 GPU/CPU benchmark 机器，见 A8）。

---

## DD-0006 · 每个数据库后端优先用自身的类型化异常分类，SQLite 例外用消息启发式

- 日期：2026-08-24 · Run: 无
- 决策：DuckDB（`duckdb.ParserException` = 语法 / `duckdb.CatalogException`
  `duckdb.BinderException` = 语义）与 PostgreSQL（`psycopg.errors.SyntaxError`
  = 语法 / `UndefinedTable` `UndefinedColumn` 等 = 语义）都有细粒度的异常类型，
  直接用 `isinstance` 分类。SQLite 的 `sqlite3` 模块语法错和语义错
  **都是** `OperationalError`，没有更细的类型，只能退回消息子串匹配
  （`"syntax error"` → SYNTAX，`"no such table"` 等 → SEMANTIC），
  并在 `classify_sqlite_exception` 的 docstring 里明写这是启发式、
  不是可信的类型判据。
- 考虑过：给 SQLite 也造一个简易 SQL 解析器先做语法检查——放弃，
  这是在重新发明 SQLite 自己的解析器，且两次解析对不上的风险
  （我们的简易解析器判"合法"但 SQLite 判"语法错"）比消息匹配更危险。
- 为什么选：消息匹配虽然脆，但每条规则都有 `tests/unit/a2/test_backends.py`
  里的真实断言撑着（不是拍脑袋写的字符串列表），且未匹配到任何已知模式时
  落到 `UNCLASSIFIED`（CLAUDE.md §2.2 允许的唯一兜底），不会被误判成
  SYNTAX 或 SEMANTIC 中的任何一个——错误分类会污染 GRPO 奖励，
  "不知道就说不知道"比"猜一个看起来对的"更安全。
- 什么情况会失效：SQLite 官方 error message 文案变化（历史上发生过），
  匹配规则要跟着版本更新；`pyproject.toml` 里没有钉死 Python 内置
  `sqlite3` 版本（它跟 Python 解释器走），这是需要在 manifest 里记录
  Python 版本的原因之一。
- 实测数字：无。
