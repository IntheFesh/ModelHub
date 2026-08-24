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
