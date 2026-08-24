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

---

## DD-0007 · Spider 难度分类器：自研近似，明确标注未与官方脚本对齐

- 日期：2026-08-24 · Run: 无（`make verify-a1` 34/34 + 3/3 元测试通过是产物）
- 决策：Spider 官方数据集本身不带 difficulty 字段，`FACTS.md` 里引用的
  easy248/medium446/hard174/extra166 是 Spider 官方 `evaluation.py`
  的 `eval_hardness` 在评测时算出来的。本沙箱没有网络，拿不到那份脚本
  做逐样本比对，所以 `data/spider_hardness.py` 按公开可查的方法论
  （WHERE/GROUP BY/ORDER BY/JOIN/OR/LIKE 计数 + 嵌套子查询 + 集合操作
  + 聚合函数等）**自己重新实现**了一版，并在模块 docstring 与函数
  docstring 里明写：这是"未经验证的近似"，不是官方脚本的精确复刻，
  **不得**在任何报告/简历里把它的输出当作官方 Spider hardness 数字引用。
- 考虑过：(a) 干脆不分类，Spider 样本的 difficulty 全部留 `None`；
  (b) 尝试从记忆里精确复刻官方脚本的每一个阈值当作"就是官方实现"。
- 为什么选：(a) 会让 A4 的"按难度分层报告"这个功能对 Spider 数据完全失效，
  价值损失比"标注清楚的近似值"更大；(b) 是本项目反作弊条款最想防的那类
  行为——把不确定当确定，把"记忆里大概是这样"包装成"官方实现"。
  选择"自研 + 显著标注 + 待真机联网验证"，是 CLAUDE.md §12
  "不确定就说不确定"的直接应用。
- 什么情况会失效：真机联网后必须拿真实 Spider dev.json 跑一遍这个分类器，
  和官方 `evaluation.py` 的输出逐条 diff；如果分布对不上
  （不是 248/446/174/166 这个已知分布），要么调整阈值，要么干脆
  换成直接 vendor 官方脚本而不是自己重写。这条 DD 在验证完成前必须
  一直挂着，不能被后续轮次悄悄"当作已解决"。
- 实测数字：无（自研分类器目前只有 5 条手造用例的单元测试，
  不是与真实 Spider 数据的对齐验证）。

---

## DD-0008 · A1 的 gold SQL 校验拒绝"悄悄跳过"：db_root 缺失时硬失败而非降级

- 日期：2026-08-24 · Run: 无
- 决策：`build_dataset_version(..., validate_gold=True)` 在 `db_root` 是
  `None` 或目录不存在时**直接抛 `FileNotFoundError`**，而不是打印一条
  warning 然后把 `gold_validation` 悄悄设成 `None` 继续跑完整个 pipeline。
  想要真的跳过，调用方必须显式传 `validate_gold=False`。
- 考虑过：`db_root` 缺失时自动降级为"跳过校验，在报告里注明"。
- 为什么选：这正是 CLAUDE.md §1.3"未测≠通过"要防的模式——一个默认参数
  `validate_gold=True` 本意是"这项检查应该跑"，如果因为环境没准备好就
  自动降级且不阻断流程，使用者很容易在 CI 或夜间队列里根本没注意到
  这项关键的数据质量检查其实一次都没真的跑过。显式抛错逼着调用方要么
  提供真实 db_root，要么用一个和"我知道我在跳过"语义完全对应的参数
  主动做这个决定，而不是环境不对就默默改变行为。
- 什么情况会失效：如果未来这个函数被大量自动化调用点使用、
  且大多数调用场景本来就不需要 gold 校验（比如只是想测 dedup 逻辑），
  这个硬失败可能显得啰嗦——那种场景下调用方应该显式传
  `validate_gold=False`，而不是指望函数悄悄替它做这个决定。
- 实测数字：无。`tests/unit/a1/test_pipeline.py::
  test_build_dataset_version_requires_explicit_opt_out_of_gold_validation`
  是这条决策的可执行证明。

---

## DD-0009 · 修正 DD-0007："无网络"的判断过早，`raw.githubusercontent.com` 实际可连

- 日期：2026-08-24 · Run: 无
- 决策：在开始 A3（比对器要用到官方 BIRD/Spider 评测脚本做一致率对照）
  之前，重新试探了一遍网络边界，发现 `huggingface.co` / 
  `extensions.duckdb.org` 走的策略代理确实拦截，但 `raw.githubusercontent.com`
  和 `api.github.com` 直接可连（200）。于是把真实的
  `taoyds/spider/evaluation.py`（Apache-2.0）拉下来，逐行核对
  `Evaluator.eval_hardness` 的四段 if/elif 阈值——**和我凭记忆写的 §DD-0007
  阈值逻辑完全一致**，但也确认了两处此前没意识到的差距：(a) 官方版本会
  把 `LIMIT` 计入 component1，我的正则版本完全没数它；(b) 官方版本靠
  `process_sql.py`（依赖 NLTK 分词 + 数据库 schema）解析出结构化 SQL 后
  数分量，我的版本是纯文本正则，对隐式 JOIN（`FROM a, b WHERE ...`）
  和字符串字面量里恰好出现 "or"/"like" 的边界情况仍然不精确。
  已修：补上 LIMIT 计数，并把 `spider_hardness.py` 的文档重写成精确区分
  "阈值逻辑：已用官方源码核实" vs "分量计数：仍是文本近似"这两层置信度，
  不再笼统说"整体未验证"。
- 考虑过：既然能连 GitHub，要不要把 `process_sql.py` 一并 vendor 进来做到
  字节级复刻？放弃——它依赖 NLTK 的 `word_tokenize`，而 NLTK 的分词模型
  数据本身还要从另一个域名（通常是 nltk 自己的 S3/GitHub Releases）单独
  下载，是否可连没有验证过，贸然引入一个新的、同样未经验证的网络依赖，
  换来的收益（分量计数从"文本近似"变成"字节级精确"）在当前阶段不成比例。
- 为什么选：这是"先怀疑自己的判断，再动手"的一个具体例子——上一条决策
  （DD-0007）里"没有网络"这个前提本身就没有被验证到位（只测了
  huggingface.co 一个域名就断言"没有网络访问真实 evaluation 脚本"），
  属于过早下结论。CLAUDE.md §12 既要求"不确定就说不确定"，
  也隐含着"要主动去核实能不能确定"，不是查一次就停。
- 什么情况会失效：如果之后确认 nltk 的分词数据其实也能下载，DD-0007/
  DD-0009 关于"分量计数仍是近似"的部分就该继续往前推进到字节级复刻，
  而不是止步于这里。
- 实测数字：无。`tests/unit/a1/test_spider_hardness.py::
  test_limit_counts_toward_component1` 是这次修正的回归测试。

---

## DD-0010 · check_no_cheating 新增窄口径 allowlist 机制，而不是把整个文件排除扫描

- 日期：2026-08-24 · Run: 无（`make verify-a3` 48/48 通过是产物）
- 决策：`compare/official_baselines.py` 需要**忠实复刻** BIRD 官方评测脚本
  的确切行为（包括它本身的缺陷：`except Exception: return 0`）用于一致率
  对照，这个模式字面上会命中 `check_no_cheating.py` 的
  `EXCEPT_RETURN_CONSTANT` 规则。没有整体跳过这个文件的扫描
  （比如加进 `.check-no-cheating-ignore` 之类的文件级豁免），而是给
  `check_no_cheating.py` 加了一个窄口径、要求写明理由的行内注释豁免：
  `# check-no-cheating: allow=<RULE> reason=<必填>`，且必须精确匹配
  被命中的那条规则名，不匹配就不生效；命中后仍然打印
  （标 ALLOWLISTED，带 reason），只是不计入退出码。
- 考虑过：(a) 整文件排除扫描；(b) 把返回值从 `0` 改成一个具名常量
  `_SCORE_ZERO = 0` 从而在 AST 层面绕开"bare Constant"匹配。
- 为什么选：(a) 会让这一整个文件失去反作弊扫描的保护——它除了这一处
  "故意复刻缺陷"之外，其它任何真实 bug（比如误加了别的 except-pass）
  都会被一起放过，代价太大。(b) 技术上能让扫描器不报，但那是在"迎合
  检查器的字面规则"而不是"表达真实意图"——本质上是一种 gaming，
  和反作弊条款的精神相反。窄口径 + 必填理由 + 精确规则名匹配，
  是唯一既不削弱扫描器、又能诚实表达"这里的例外是有意的、有记录的"
  的做法。顺手还用同样的思路给 `_example_rows`（截断调试用的差异行样例，
  不是截断真实比对数据）加了一处豁免，并把截断上限显式写回返回的
  data 里（呼应 `ResultSet.truncated` 的模式），而不是只靠注释说明。
- 什么情况会失效：如果 allowlist 注释本身开始被滥用（比如理由写得很敷衍），
  需要人工 code review 把关——这个机制只保证"豁免是显式的、可 grep 的"，
  不保证"每个豁免理由都站得住脚"，后者仍然需要人读。
- 实测数字：无。`tests/meta/a0/test_check_no_cheating.py` 里
  `test_allowlist_comment_suppresses_the_named_rule_only` /
  `test_allowlist_comment_does_not_suppress_a_different_rule` /
  `test_allowlisted_finding_does_not_fail_main_exit_code` 三条元测试
  覆盖了这个机制本身能不能正确工作（包括"规则名对不上就不生效"这个
  防滥用的关键行为）。

---

## DD-0011 · 用真实的 BIRD 官方评测脚本做一致率对照，而不是假想的"官方脚本"

- 日期：2026-08-24 · Run: 无
- 决策：`compare/official_baselines.py` 是从
  `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`（MIT License）
  真实抓取下来、逐行核对过的 `execute_sql` 函数复刻，不是凭印象写的
  "差不多类似 BIRD 做法"的代码。一致率脚本
  （`compare/consistency.py`）拿这份真实复刻和我们自己的比对器对照，
  产出的分歧案例是真实分歧，不是编出来演示"我们比官方更严谨"的样例。
  这也顺带核实了 FACTS.md/A3 面试预期问题"为什么不直接用官方脚本当
  GRPO 奖励函数"的具体答案：官方脚本用 `set()` 比较——无浮点容差、
  重复行被折叠成集合、行序永远不检查、任何异常一律记 0（分不清语法错/
  语义错/超时/我们自己的 harness 挂了）。这些不是猜测，是读源码读出来的。
- 考虑过：凭对 BIRD 论文/公开报告的印象写一个"差不多"的官方对照实现。
- 为什么选：CLAUDE.md §12 的"不确定就说不确定"反过来也要求"确定的时候
  别装作不确定"——本 session 已经证实 `raw.githubusercontent.com`
  可连（DD-0009），继续凭记忆编一个"模拟官方脚本"而不是抓真实源码，
  就是在明知可以核实的情况下选择不核实，这本身就是一种不诚实。
- 什么情况会失效：DAMO-ConvAI 仓库后续更新了 `evaluation.py`
  的比较逻辑（比如加了浮点容差），这份 vendored 复刻就会和最新官方版本
  脱节，需要重新抓取核对——`official_baselines.py` 的 docstring 里
  记了抓取日期（2026-08-24），方便以后判断是否该重新核对。
- 实测数字：无。`tests/unit/a3/test_official_baselines.py::
  test_bird_official_compare_collapses_duplicates_unlike_our_comparator`
  是"官方脚本折叠重复行"这条结论的可执行证明，不是转述。

---

## DD-0012 · eval_tier 的归属：run_eval 不接受也不丢弃它，report.py 才是唯一权威

- 日期：2026-08-24 · Run: 无（A4 本轮不产生真实性能/准确率数字，
  见下方"实测数字"）
- 决策：`eval/runner.py::run_eval` 最初的签名里有一个 `eval_tier` 参数，
  写完才发现它在函数体内完全没被使用——`run_eval` 到底跑的是 quick
  还是 full，是"调用方传了哪份 `samples` 列表"这个事实本身决定的，
  不是一个需要单独再传一次、且传错也不会被这层代码发现的字符串标签。
  把这个参数删掉了：`run_eval` 只管生成/执行/比对/落盘，"这批预测属于
  哪个 tier"这件事完全交给 `RunManifest.eval_tier`（写 manifest 时由
  调用方显式指定）和 `eval/report.py`（`REQUIRED_MANIFEST_FIELDS` 里
  硬性要求 `eval_tier` 非空，且报告第一行就把 tier 显式印出来）。
- 考虑过：(a) 保留 `eval_tier` 参数，函数体内什么也不做，只是"因为以后
  可能要用"预留一个位置；(b) 让 `run_eval` 自己校验 `eval_tier` 和
  `samples` 数量/来源是否匹配（比如 quick 必须恰好 500 条固定子集）。
- 为什么选：(a) 是"接受了却立刻丢弃"的典型坏味道——一个参数如果函数体
  不读它，签名里出现它本身就是一种误导（调用方会以为传了就生效）。
  (b) 把校验放错了层：`run_eval` 是一个不知道"BIRD Mini-Dev V2 的
  500 条固定子集"这种数据集特定事实的通用执行器，让它去校验"这是不是
  合法的 quick-tier 子集"会把 A1（data/）的知识硬编码进 A4（eval/），
  违反 CLAUDE.md §9"禁止跨层反向 import"背后的分层原则。真正需要
  "这份报告到底是哪个 tier"这个事实的唯一消费者是报告本身，
  所以让 `report.py::assert_report_ready` 在渲染报告之前对
  `manifest.eval_tier is None` 硬失败，比在更早的执行层加一个用不上的
  参数更符合"字段该在谁的层里被强制"的原则。
- 什么情况会失效：如果未来出现"运行时就要按 tier 切换行为"的真实需求
  （比如 quick-tier 允许更激进的并发度、full-tier 必须限流保护
  eval DB），`run_eval` 才需要真正读这个参数而不是转发它——到那时候
  这个决策要重新评估，不是简单地把参数加回来。
- 实测数字：无。`tests/meta/a4/test_report_rejects_missing_metric.py::
  test_report_rejects_missing_eval_tier` 是"tier 缺失时报告生成器硬失败"
  这条结论的可执行证明。

---

## DD-0013 · GenerationResult 用 ModelHubBaseConfig，不是裸 pydantic.BaseModel

- 日期：2026-08-24 · Run: 无
- 决策：`eval/model_client.py::GenerationResult` 最初直接继承
  `pydantic.BaseModel`，没有走项目统一的 `ModelHubBaseConfig`
  （`ConfigDict(extra="forbid", frozen=True, validate_default=True)`）。
  这是本轮写
  `tests/unit/a4/test_model_client.py::test_generation_result_forbids_unknown_field`
  时被测试真实跑挂发现的——传一个拼错的多余字段进去，裸 `BaseModel`
  默认 `extra="ignore"`，不报错也不提示，静默吞掉。这正是
  CLAUDE.md §4 点名的"拼错的 key 必须报错"要防的那类问题：如果
  `HttpModelClient.generate()` 解析真实 vLLM 响应时字段名对不上
  （比如 vLLM 版本升级后 usage 字段改名），裸 BaseModel 会静默丢字段
  而不是在开发阶段就暴露出来。改成继承 `ModelHubBaseConfig`，
  同一个测试从失败变成通过。
- 考虑过：无——这不是一个有多个合理选项的架构决策，是一处不该发生的
  疏漏，唯一正确做法就是改成项目统一基类。之所以单独记一条 DD，
  是因为它是"写测试而不是写完就自认为对"这条工作方式在本轮抓到的
  一个真实、具体的例子，值得留痕而不是悄悄改掉。
- 什么情况会失效：不适用（bug 修复，不是会过期的架构假设）。
- 实测数字：无。测试本身就是证据：
  `tests/unit/a4/test_model_client.py::test_generation_result_forbids_unknown_field`
  修复前失败（`DID NOT RAISE ValueError`），修复后通过。

---

## DD-0014 · A5 的范围划定：只做"服务层需要知道的静态事实"，不启动真实 vLLM

- 日期：2026-08-24 · Run: 无（A5 不产生性能数字，见下方"实测数字"）
- 决策：`serve/` 本轮只做四件事——(1) 从真实 SQLite 文件反射出 schema 并
  渲染成 prompt 文本（`schema_format.py`/`prompt.py`，直接补上 A4/DD-0012
  留空的 `prompt_builder` 插槽）；(2) GDN 快 kernel 生效检测
  （`kernel_status.py`，把 `preflight.py` 里人工写好的 G1/G2 真实 CUDA
  kernel 调用逻辑重构成可被 serve 时/A9 门禁复用的函数，不是重新发明一遍）；
  (3) 双模型的 KV 经济学（`model_profile.py` + `configs/serve/model_profiles/
  *.yaml`，公式可独立复现 FACTS.md 表里的 56 KiB/32 KiB）；(4) vLLM 启动
  日志解析（`vllm_log_parser.py`）。**没有**实现"真的启动一个 vLLM 进程
  并服务请求"——那需要真实 GPU + 下载两个模型权重，本沙箱都没有。
- 考虑过：(a) 把"启动 vLLM 子进程、暴露 HTTP 接口"也一起写掉，用
  `subprocess.Popen(["vllm", "serve", ...])` 拼出完整命令，反正沙箱里
  跑不通就标 `requires_gpu` 跳过——这正是 A4 处理 `HttpModelClient` 时
  用过的模式（DD-0003）。
- 为什么不跟着 (a) 做：`HttpModelClient` 那次能这么做，是因为"发一个
  HTTP 请求、解析 JSON 响应"这段逻辑本身在没有服务端时也是可读、可 review
  的完整代码，唯一缺的是运行时对端。而"拼出 `vllm serve` 的启动命令行"
  这件事的正确性高度依赖 vLLM 当前版本的真实 CLI 参数名
  （FACTS.md 自己都点名"vLLM 当前版本的 MTP/推测解码参数名"是需要
  `vllm serve --help` 核对的未知项，见 FACTS.md 表格第 6 行）——在没有
  网络装 vLLM、没有 GPU 验证任何一个参数名的情况下写这段代码，
  不是"写了但未验证的真实代码"，而是更接近凭印象编一个"像是"正确的
  命令行，这正是 CLAUDE.md §12"不确定就说不确定"要挡的那类东西。
  schema/prompt/kernel-detection/log-parser 这四块则不同：
  schema 反射和渲染是纯逻辑，在真实 SQLite 文件上完整可测；kernel
  detection 是把已经存在于仓库里、人工写好并且格式明确的 `preflight.py`
  G1/G2 原样复用；KV 公式是从 FACTS.md 已经写死的架构事实（层数/KV头/
  head_dim）反推，可独立核对；日志解析器则明确把"确信的 pattern"
  （`# GPU blocks:`、`Maximum concurrency for`、`Loading model weights
  took`）和"不确信的 pattern"（GDN 状态占用那一行的具体格式）分开标注，
  没有把两者混为一谈。
- 什么情况会失效：一旦用户在真实 GPU 机器上跑通了 `vllm serve` 并把
  真实启动命令/参数回传，"拼命令行启动 vLLM"这部分工作应该补上——
  到那时候有真实命令行可以核对，不再是凭印象编。
- 实测数字：无。`tests/unit/a5/test_vllm_log_parser.py` 里的日志片段是
  手写的、按已知 vLLM 日志格式构造的，不是真实 run 的产物，
  docstring 里写明了这一点。

---

## DD-0015 · gateway/ 的限流与配额用真实 Redis + 原子 Lua 脚本，不是"先 GET 后 SET"

- 日期：2026-08-24 · Run: 无（A6 不产生性能数字）
- 决策：`rate_limit.py`/`quota.py` 的计数器递增 + 首次设置过期时间，
  写成单条 `EVAL` 原子脚本（`INCR`/`INCRBY` 之后判断是否要设 `EXPIRE`），
  不是两次独立的 Redis 往返。本沙箱装了真实 Redis（`redis-server` 已经
  在系统里，只是没启动——启动后 `redis-cli ping` 真实返回 `PONG`），
  所以测试对着真实 Redis 跑，不是拿 Python dict 假装一个 Redis。
- 考虑过：(a) `INCR` 后单独一次 `EXPIRE` 调用（更直观，两行代码）；
  (b) 用 `redis-py` 的 `pipeline(transaction=True)`（MULTI/EXEC）代替
  Lua 脚本。
- 为什么选 Lua 脚本而不是 (a)：两次独立往返之间存在真实的竞态——
  如果进程在 `INCR` 和 `EXPIRE` 之间崩溃，或者两个并发请求交错执行，
  会留下一个永远不过期的计数器键（配额/限流的失效模式是"多算了一点点"
  还是"这个租户从此再也限不住"，后者是本项目不能接受的）。
  为什么选 Lua 脚本而不是 (b)：MULTI/EXEC 保证的是"这批命令按顺序、
  不被其他客户端命令插入执行"，但它本身不支持"读取 INCR 的返回值来
  决定要不要执行 EXPIRE"这种条件逻辑（pipeline 里的命令是无条件打包
  发送的，看不到前一条命令在服务端的执行结果）——真要用 MULTI/EXEC
  实现"仅在计数器刚创建时才设置 TTL"，需要 WATCH/重试循环，比一条
  Lua 脚本更复杂且不天然更安全。Lua 脚本在 Redis 服务端单线程执行，
  是这个需求最简单也最正确的写法。
- 什么情况会失效：如果未来切到 Redis Cluster 且限流键分布在不同分片，
  单条 Lua 脚本仍然安全（脚本只碰一个 key，符合 cluster 的单 slot
  限制），但如果哪天需要跨多个 tenant key 做一次原子操作，这个模式
  就不再直接适用，需要重新设计。
- 实测数字：无。`tests/meta/a6/test_rate_limit_and_quota_are_atomic_under_concurrency.py`
  用真实 50 线程并发对同一个 key 压 `check_rate_limit`/
  `record_and_check_quota`，断言放行数量精确等于配置的上限——这条测试
  验证的是"限流器/配额跟踪器在真实并发下确实精确卡在配置的上限"这件事
  本身（一个值得单独用真实并发验证、而不是只靠读代码相信的正确性
  性质），**不是** Lua 脚本这个实现选择本身的回归测试：Redis 的
  `INCR`/`INCRBY` 单条命令天然原子，即便写成"先 INCR 再单独一次
  EXPIRE"两次往返，计数本身大概率也不会在并发下失准——没有实际写
  两步版本去验证过，不应该、也没有声称这条测试会因为那个改动而变红。
  Lua 脚本真正解决的是 §"为什么选"里写的那个问题：进程在 INCR 和
  EXPIRE 之间崩溃/被杀，会留下一个永不过期的计数器键——这是一个
  时序窗口很窄、用普通测试基本测不出来的故障模式，本条目诚实地
  承认它没有可执行验证，只有推理依据。

---

## DD-0016 · gateway 的路由阈值故意留 estimate，不在这一轮编一个"测出来"的数字

- 日期：2026-08-24 · Run: 无
- 决策：`configs/gateway/routing.yaml` 的 `schema_token_threshold: 1500`
  配一个 `threshold_source: "estimate"` 字段，明确标注这不是 A8 压测出的
  真实交叉点，只是 FACTS.md 文档里"1k–3k 之间交叉"这个区间的一个居中
  取值。`RoutingConfig` 的 schema 把这个字段设成必填（不是默认值+可选），
  逼着任何写这份 YAML 或读这份 YAML 的人都要显式面对"这个数字的来源
  是什么"这个问题。
- 考虑过：直接编一个"看起来更精确"的数字（比如 1847）让它显得像是
  测出来的。
- 为什么不这么做：CLAUDE.md §3.2"文档禁止手写数字"和 §12"不确定就说
  不确定"两条原则合在一起，禁止的不只是"编一个假数字"，也包括"用一个
  精度虚高的数字暗示它是测出来的"——1500 这种整百数字反而更诚实地
  传达"这是拍出来的中间值，不是测量结果"。等 A8 真的跑完 prompt 长度
  扫描、找到真实交叉点，`threshold_source` 改成 `"measured"`，
  数值本身也替换掉，两个改动一起发生，不会出现"数值已经很精确但
  source 字段忘了改"的不一致状态（因为两者本来就该在同一次改动里完成）。
- 什么情况会失效：如果 A8 的压测发现交叉点其实在 1k 以下或 8k 以上
  （比 FACTS.md 估计的范围偏差很大），说明前面对 KV 经济学的估算本身
  有问题，需要回头核对 `serve/model_profile.py` 的层数/KV头/head_dim
  配置，而不是简单地把这个数字覆盖掉了事。
- 实测数字：无。`tests/unit/a6/test_routing.py::
  test_real_routing_config_loads_and_is_labeled_an_estimate` 断言
  真实提交的配置文件里 `threshold_source == "estimate"`，防止有人
  以后不小心把这个字段悄悄改成 `"measured"` 却没有真的跑 A8。

---

## DD-0017 · MFU/MBU 计算函数不接受"没测就用规格值"的隐式兜底

- 日期：2026-08-24 · Run: 无（A7 不产生真实性能数字）
- 决策：`monitor/mfu_mbu.py::compute_mfu`/`compute_mbu` 的
  `peak_bf16_flops`/`peak_bw_bytes_s` 是必填参数，函数内部没有
  "调用方不传就用 209 TFLOPS / 1792 GB/s 规格值"这种默认值。真正的
  测量函数（`hardware_bench.py::measure_bf16_tflops`/
  `measure_memory_bandwidth_gbs`，从 `preflight.py` E1/E2 的真实
  GEMM/memcpy 微基准重构而来）在测不出时返回 `status=SKIP` 和
  `tflops=None`/`gbs=None`，调用方必须显式决定"没测出来就用文档估算值"
  还是"没测出来就拒绝算 MFU/MBU"——这个决定不能被函数悄悄替调用方做了。
- 考虑过：给 `compute_mfu`/`compute_mbu` 加 `peak_*: float | None = None`
  加上"None 时退回 209/1792"的兜底逻辑，调用更省事。
- 为什么不这么做：这正是 FACTS.md 自己反复强调的红线——"未校准"的算力/
  带宽假设一旦被悄悄代入公式，算出来的 MFU/MBU 数字会看起来完全正常
  （不会报错、不会是明显异常值），但已经不再是"这次运行实测的瓶颈画像"，
  而是"文档估算值和真实吞吐量拼出来的四不像数字"。manifest 里专门留了
  `measured_peak_tflops`/`measured_bw_gbs` 两个字段（A0 就定义好了），
  就是为了让"这个 run 的峰值到底是测出来的还是假设的"这件事可追溯——
  如果 MFU/MBU 函数自己悄悄兜底，这两个字段的意义就被绕过去了。
- 什么情况会失效：如果未来某个场景下"用文档估算值算一个大概的 MFU/MBU
  仅供排程参考"是合理需求（不是对外报告用），应该在调用方那一层
  显式写清楚"这是估算值"，而不是改这两个函数本身加隐式默认——
  调用方的显式选择永远应该留痕。
- 实测数字：无。`tests/meta/a7/test_unmeasured_hardware_never_fabricates_a_manifest_value.py`
  证明本沙箱真实测不出峰值时（`torch` 确实没装），这两个测量函数的
  `None` 结果会原样传进 `RunManifest`，不会被替换成任何数字。

---

## DD-0018 · GDN 固定态是"每并发请求"成本，不是"全卡一次性"成本——一个真实算错的 bug

- 日期：2026-08-24 · Run: 无（A8 不产生真实压测数字）
- 决策：`bench/capacity_planning.py::estimate_max_concurrent_requests`
  把 `ModelProfile.gdn_fixed_state_bytes` 计入**每个并发请求**的显存开销
  （`per_request_bytes = kv_bytes_per_token × (prompt+max_new) +
  gdn_fixed_state_bytes`），而不是从整卡的共享可用预算里只扣一次。
- 考虑过：最初的实现（写 A8 第一版时）把它当成"全卡只有一份"从
  `usable_bytes` 里扣一次——这是 A5 写 `gdn_fixed_state_bytes` 字段时
  文档措辞"per-layer fixed-size recurrent state, independent of
  sequence length"埋下的歧义："跟序列长度无关"没错，但没说清楚
  "跟并发请求数有没有关"，实现时想当然按"全局一份"处理了。
- 怎么发现的：给 `find_concurrency_crossover_point` 写单元测试时，
  想用真实的 Arctic-7B / Qwen3.5-9B 配置文件复现
  MODEL-SELECTION-FINAL.md 明确写的"曲线在 1k–3k 之间交叉"这个结论——
  结果按"全局扣一次"的公式算出来，Qwen3.5-9B 在**任何** prompt 长度下
  并发数都比 Arctic-7B 高，根本不存在交叉点，跟文档正面矛盾。
  GDN 的循环状态（recurrent state）本质上和 KV cache 一样是**每个在途
  生成序列各自一份**——两个并发请求各自需要一份独立的状态矩阵，
  不是整张卡共享一份。把这个开销从"每张卡扣一次"改成"每个并发请求都要
  掏"之后，用真实配置文件重算：Arctic 在 1k token 时并发数反超 Qwen3.5
  （214 vs 183），到 3k 时被反超（71 vs 80）——跟文档"1k–3k 之间交叉"
  的说法几乎精确吻合。
- 为什么这是可信的修复而不是"调参数凑答案"：改动前后用的都是同一份
  真实 YAML 配置（`configs/serve/model_profiles/*.yaml`，A5 就已经
  写死、这一轮没有改动任何数值），改的只是"这个字节数该被谁承担"这一
  处代码逻辑；改完之后不仅方向对了，连"1k-3k 之间"这个区间都对上了，
  这是模型语义修对了带来的自然结果，不是为了让测试变绿反向调整了
  测试数据（`tests/meta/a8/test_gdn_per_request_cost_matches_documented_crossover.py`
  把错误公式原样保留在测试文件里做对照，任何人都能重新验证这个结论）。
- 什么情况会失效：如果真实 vLLM/GDN 实现里循环状态其实是可以跨请求
  共享的（比如某种 batched 状态复用优化），这个"每请求一份"的假设
  就需要重新核实——目前没有证据支持"可共享"这个更乐观的假设，
  按更保守（每请求一份）的假设做容量规划更安全。
- 实测数字：无。`tests/unit/a8/test_capacity_planning.py::
  test_real_arctic_vs_qwen_profiles_reproduce_the_documented_crossover`
  和上面提到的元测试都是可执行验证，不是叙述。

---

## DD-0019 · 五道闸各自独立跑完，不因为前面一道拒绝就短路

- 日期：2026-08-24 · Run: 无（A9 不产生真实门禁数字，本沙箱没有真实
  候选模型可跑）
- 决策：`gate/admission.py::run_admission_gate` 无论前面哪道闸已经
  REJECT，五道闸（pollution/accuracy/truncation/safety/regression）
  全部跑完，`GateVerdict.results` 里每一道都有结果，不是命中第一个
  REJECT 就提前返回。
- 考虑过：像大多数校验管线一样，第一道不过直接短路返回，省掉后面几道
  的计算。
- 为什么选：PLAN.md 明确写了门禁演练要验证"ckpt-A/B/D 被不同闸拦下"——
  这个演练的价值恰恰在于"看清楚每一个坏 checkpoint 具体是被哪一道拦下
  的"，如果 A 型坏模型（欠拟合）在准确率闸就短路返回，团队永远不会
  知道它在安全闸/截断率闸上表现如何，也没法在一次门禁运行里同时验证
  "这个 checkpoint 是不是恰好也在别的维度上有问题"。短路优化省的是
  可忽略的计算量（本项目量级下五道闸都很便宜），换来的是诊断信息的
  损失，不值得。
- 什么情况会失效：如果未来某道闸本身开销很大（比如真的要跑一次完整
  700B 参数模型推理），"全部跑完"的代价可能变得不可接受，需要重新
  权衡是否要短路——但即便那时，更合理的做法也是把开销大的闸放到最后
  一个跑，而不是恢复"命中即返回"。
- 实测数字：无。`tests/unit/a9/test_admission.py::
  test_low_accuracy_candidate_is_rejected_and_other_gates_still_run`
  验证了 accuracy 闸 REJECT 之后，safety 闸依然真实执行并给出了
  PASS——不是因为超时/异常侥幸跑到了，是设计如此。

---

## DD-0020 · 安全闸测的是沙箱的只读保证，不是候选模型的输出

- 日期：2026-08-24 · Run: 无
- 决策：`gate/safety_gate.py::check_safety_gate` 直接执行 Mini-Dev V2
  270 条 CRUD 对抗样本各自的 `gold_sql`（真实的 DELETE/UPDATE/INSERT/
  DROP 语句）通过真实的 `sqlexec.execute_isolated`，而不是先让候选
  模型针对这些问题生成 SQL、再检查模型生成的 SQL 有没有被拦下。
- 考虑过：跑模型生成 + 检查生成结果——这样"更像"是在评估模型行为。
- 为什么不这么做：`data/safety_adversarial.py`（A1）的 docstring
  原话是"a BI/analytics platform should reject 270/270 of these
  outright"——这句话的主语是**平台**，不是模型；而且 A2 的
  sqlexec 已经用只读连接从机制上保证任何真实写操作永远无法成功
  （不管模型说什么），所以"检查模型生成的 SQL 有没有被拦下"这件事
  的答案在系统设计层面已经是确定的"永远会被拦下"，拿它去评估模型
  没有实际信息量。真正有价值、且会随时间变化（比如换了 sqlexec 后端、
  改了连接参数、引入新 backend）的是"沙箱自身的只读保证是否还成立"——
  这才是需要在每次模型上线前重新验证一遍的东西，跟具体候选模型是
  哪一个无关。
- 什么情况会失效：如果未来平台允许模型直接绕过 sqlexec（比如接入一个
  不走本项目沙箱的外部执行通道），这道闸就测不到那条路径，需要针对
  新通道单独补一道类似的闸，而不是简单复用这一个。
- 实测数字：无。`tests/unit/a9/test_safety_gate.py::
  test_data_actually_unmodified_after_blocked_attempts` 在真实
  SQLite 上验证了"闸判定为 PASS"和"数据真的没被改"这两件事一致——
  不是只信任 exec_code，而是回头查了一遍表内容。

---

## DD-0021 · 灰度"到达 100%"和"完成"是两个不同的状态，不能合并

- 日期：2026-08-24 · Run: 无（A10 不产生真实灰度数字，本沙箱没有真实
  线上流量可观测）
- 决策：`release/canary.py::try_advance_stage` 把"当前档已经在最后一档
  （100%）"和"可以标记为 COMPLETED"分成两次独立判断：即使
  `stage_index` 已经指向最后一档，也仍然要求在**这一档**再观测满
  `min_requests_per_stage` 个请求，才会把状态从 `IN_PROGRESS` 翻成
  `COMPLETED`。反映到编排层：`orchestrator.py::run_canary_cycle`
  在"从 50% 推进到 100%"和"100% 观测完毕、正式收尾"之间，需要两轮
  独立的健康检查周期，不是一轮。
- 考虑过：把"stage_index 走到 len(stages)-1"本身当作完成信号，推进到
  100% 的同一次调用里直接返回 COMPLETED，省一轮健康检查。
- 为什么选：100% 阶段本身就是灰度要保护的对象之一——如果"推进到
  100%"和"验证过 100%"是同一个动作，那么"在 100% 阶段观测到的健康
  数据"就永远不会被用来做回滚判断，100% 这一档的自动回滚保护形同
  虚设。PLAN.md 的"灰度 5% → 注入劣化 → 自动回滚"drill 没有明确排除
  劣化发生在最后一档的场景，而这恰恰是影响面最大的一种（100% 流量
  已经在受影响）。
- 什么情况会失效：如果未来的运维流程认为"100% 只是一个终点标记，不
  需要单独的健康周期"（比如上游已经有独立的、更细粒度的 100% 监控
  覆盖这段时间），可以重新合并两个状态——但要显式记录这个权衡，不能
  悄悄改行为。
- 实测数字：无。写测试时最初按"一次调用即完成"的假设写的
  `tests/unit/a10/test_canary.py::
  test_try_advance_stage_completes_at_last_stage` 和
  `tests/unit/a10/test_orchestrator.py::
  test_healthy_traffic_at_last_stage_completes` 两条断言都跑红了——
  照实现代码的真实行为改的测试，不是反过来改实现去迁就一开始写错的
  断言。

---

## DD-0022 · 线上抽样"成功"的定义就是 sqlexec 自己的 EXEC_OK，不做二次筛选

- 日期：2026-08-24 · Run: 无
- 决策：`release/online_sampling.py::predictions_to_health_samples` 把
  `succeeded` 定义为 `prediction.exec_code is ErrorCode.EXEC_OK`，
  逐条覆盖 SYNTAX/SEMANTIC/TIMEOUT/OUTPUT_TRUNCATED/
  HARNESS_DB_UNAVAILABLE/HARNESS_INTERNAL 全部计为"不成功"（`
  is_harness_error` 单独另算，供 `RollbackConfig.
  max_harness_error_rate` 区分"系统错"和"模型错"两类阈值）。
- 考虑过：写这个函数的第一版时，`succeeded` 定义成了
  `not p.is_harness_error and p.exec_code.value != "SYNTAX"`——只排除
  语法错误，语义错误/超时之类没考虑，是写的时候顺手拍的一个不完整
  条件，不是深思熟虑的选择。
- 为什么改：CLAUDE.md §2.3 的六分类里，SYNTAX/SEMANTIC/TIMEOUT 三类
  都是"模型错，reward = 0"这一档——线上灰度的画布上没有理由只挑
  SYNTAX 特殊对待。一个金丝雀版本如果开始比稳定版本产生更多的
  SEMANTIC 错误（比如换了新 schema 之后模型引用了不存在的列）或者
  TIMEOUT（比如新版本生成的 SQL 更容易触发笛卡尔积），这本身就是需要
  触发自动回滚的真实信号，不应该被排除在"成功"判定之外。
- 什么情况会失效：如果未来产品决定某些错误类别对线上健康判定不重要
  （比如 OUTPUT_TRUNCATED 已经在别处单独监控、不该重复计入这里的
  error_rate），需要显式把它从"不成功"里摘出来、并在这里写清楚原因，
  不能悄悄放宽这个定义。
- 实测数字：无。`tests/unit/a10/test_online_sampling.py::
  TestPredictionsToHealthSamples::
  test_syntax_error_counts_as_not_succeeded_but_not_harness_error`
  和同文件里 harness-error 的对照测试，一起验证了"模型错"和"系统错"
  在这个转换里被分别记到了 `succeeded` 和 `is_harness_error` 两个
  独立字段上，不会互相污染。
