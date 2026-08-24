# ModelHub · Build Log

格式见 `PLAN.md` §十。本沙箱无 GPU，训练相关字段大量标注"待真机"。

---

```
D0 2026-08-24（沙箱 session，非真实 GPU 机器）
做了什么：
  - Round 0：仓库骨架、pyproject.toml（ruff+mypy strict+pytest）、Makefile
    的通用 verify-<id> 目标。make lint / make typecheck 在空骨架上通过。
  - A0：common/ 地基层——errors.py（ModelHubError + 扁平 ErrorCode，
    SQL 七分类 + UNCLASSIFIED）、config.py（Pydantic v2 extra=forbid +
    config_hash）、logging.py（结构化 JSON + 敏感信息脱敏 + truncate_for_log
    与真实数据截断路径分离）、run_manifest.py（RunManifest + 三污染位
    + assert_not_polluted 硬失败网关）、atomic_io.py（.tmp→os.replace 原子写）、
    capability.py（CheckStatus + is_available 白名单门，供全项目复用）、
    scripts/check_no_cheating.py（AST 扫描 8 类反作弊模式）、
    scripts/check_placeholders.py。61 个单元/元测试全绿，
    check_no_cheating 扫描 src/ 本身也是 clean。

遇到什么问题：
  1. 环境探测：本容器 `nvidia-smi` 不存在、huggingface.co 经代理返回 403、
     可写磁盘仅 30GB —— 确认无法真实下载 BIRD/Spider/HF 模型权重、
     无法起 vLLM、无法训练。PyPI 与 apt 主源可直连，可以装真实的
     SQLite/DuckDB/PostgreSQL/Redis/FastAPI 等做 CPU-only 层的真实集成测试。
  2. `scripts/check_no_cheating.py` 里 `ast.Slice` 节点我最初写成
     `node.slice.stop`，实际 Python `ast` 模块里字段名是 `lower/upper/step`，
     不是 `stop` —— 运行时 `AttributeError`，被冒烟跑一次自己就暴露了。
  3. `atomic_io.atomic_write_bytes` 最初把 `path.parent.mkdir(...)`
     放在 try 块外面，父路径实际是个文件（而非目录）时会抛出未分类的
     `NotADirectoryError` 而不是 `ModelHubError`——被
     `test_atomic_write_bytes_rejects_unwritable_target` 抓到，是先写测试
     再写实现暴露出的一个真实边界 bug，不是编出来演示流程的假 bug。
  4. cleanup 路径里最初写 `except OSError: tmp.unlink(missing_ok=True)`
     失败又抛出第二个 OSError，会掩盖原始错误；同时这种写法在结构上
     和 `check_no_cheating.py` 要拦的 EXCEPT_PASS 模式很接近。改用
     `contextlib.suppress(OSError)` 包裹 best-effort 清理，
     既避免掩盖原始异常，又不落入反作弊扫描要拦截的 except-pass 形态
     （这不是绕过检查——是两种写法在语义上确实不同：suppress 明确表达
     "这段清理允许失败"，而裸 except-pass 是"吞掉所有信息"）。

怎么解决的：见上。

测出什么数字：无（本轮不产生性能/准确率数字）。
```

```
D0 续（同一 session）
做了什么：
  - 装了本地 PostgreSQL 16（apt，真实跑通，非 mock）+ 建了测试角色/库；
    验证 duckdb / psycopg 均可 pip 装且能用。
  - A2：sqlexec/ —— outcomes.py（ExecOutcome/ResultSet，截断显式标记）、
    backends.py（SQLite/DuckDB/PostgreSQL 三后端的只读连接 + 执行 + 异常分类，
    只读靠数据库引擎自身强制而非正则黑名单）、sandbox.py（每任务一个
    spawn 子进程 + wall-clock 超时 SIGKILL + RLIMIT_AS 内存上限）、
    pool.py（ThreadPoolExecutor 限制并发子进程数的批量执行）。
    37 个单元/元测试全绿，覆盖真实 SQLite + 真实 DuckDB + 真实本地 PostgreSQL。

遇到什么问题：
  1. DuckDB 的 `INSTALL sqlite; LOAD sqlite;`（用来 ATTACH BIRD 的 .sqlite
     文件）需要联网下载 sqlite_scanner 扩展，本沙箱代理对
     `extensions.duckdb.org` 返回 403 —— 这条路径标 `requires_network`，
     手动临时放开验证过确实只因为网络被拦（错误分类正确落在
     HARNESS_DB_UNAVAILABLE），不是代码问题；真机联网后这条路径要重跑确认。
  2. 我自己的 `check_no_cheating.py` 在 A2 代码上抓到两个真问题（不是误报）：
     一是 `sandbox.py` 里 RLIMIT_AS 的 best-effort 兜底又写成了
     `except (ValueError, OSError): pass`，和 A0 遇到的一模一样的模式，
     同样改成 `contextlib.suppress`；二是发现规则本身有个真误报——
     `psycopg.connect(dsn, connect_timeout=5)` 明明给了超时，
     但检查器只认字面的 `timeout=` 关键字，不认 `connect_timeout=`。
     这是检查器自己的 bug，改成匹配任意包含 "timeout" 的关键字名，
     并把这个案例写成元测试锁住（不然以后又会因为同一个原因被误伤）。
  3. 故意构造一个非法 backend 值来验证"worker 进程崩溃时父进程能否正确
     报 HARNESS_INTERNAL"，结果连带发现父进程自己的兜底分支里
     `ref.backend.value` 会在 `ref.backend` 不是真正的 Backend 枚举时炸掉——
     测试把这条防御性代码里的防御性代码本身的 bug 也炸出来了，改用 `str()`。

怎么解决的：见上。

测出什么数字：无（本轮不产生性能数字；进程级 kill 的正确性由
`test_cpu_bound_cartesian_product_is_actually_killed_not_just_reported_slow`
证明，不是数字，是行为断言）。
```

```
D0 续续（同一 session）
做了什么：
  - A1：data/ —— schema.py（NormalizedSample 统一 schema，BIRD/Spider
    难度词表不强行对齐）、normalize.py（BIRD/Spider/Mini-Dev V2 三源归一化，
    缺必填字段直接 KeyError 不静默）、dedup.py（问题级去重+SQL级去重计数，
    只丢问题级重复）、hashing.py（question_hash/sql_hash/dataset_content_hash，
    忽略大小写与空白但内容变了 hash 就变）、splits.py（(question_hash, db_id)
    双键判定 train/dev/test 交集，命中即硬失败）、spider_hardness.py
    （自研难度分类器，见 DD-0007，明确标注未验证）、gold_validation.py
    （用 A2 的 sqlexec 真跑 gold SQL，失败样本逐条落盘不是只给个数字）、
    quick_eval.py / safety_adversarial.py（500 条快评 + 270 条 CRUD
    对抗集导出，互相校验 source 不能混）、pipeline.py（整轮编排 +
    DataBuildReport）、downloader.py（真实 httpx/huggingface_hub 下载逻辑，
    本沙箱验证不了，诚实标 requires_network）。37 个单元/元测试全绿
    （2 个 requires_network 被显式 deselect，不是静默跳过）。

遇到什么问题：
  1. 一开始把"gold SQL 执行校验"放在 A1 里自己重新实现一个简化执行器，
     写到一半意识到这和 A2 要造的沙箱在只读/超时/错误分类上是同一件事，
     两份实现会在语义上飘走——改成让 A1 依赖 A2 的 sqlexec 公开 API，
     并把 A2 提前到 A1 之前实现（DD-0004）。
  2. Spider 官方数据集不带 difficulty 字段，且我记忆里对官方
     `evaluation.py` 的确切阈值没有 100% 把握——没有选择"编一个看起来对的"
     或者"假装记得很准"，而是自研一版并在代码和决策记录里反复标注
     "未验证、不得对外引用"（DD-0007）。这是本轮最典型的一次
     "不确定就说不确定"实践，而不是一次技术判断。
  3. 写 `test_query_with_where_is_at_least_medium` 时断言错了——以为
     "带 WHERE 就至少 medium"，实际按我实现的（也是 Spider 真实）方法论，
     单个 WHERE 条件、没有别的复杂度分量，就是 easy。测试失败后确认
     是测试的假设错了，不是分类器错了，改的是测试断言，不是放宽分类器
     的判据去迁就一个错误的预期。
  4. `build_dataset_version` 最初设计成 `db_root` 缺失时自动跳过 gold
     校验并在报告里注明——落笔时意识到这正是 CLAUDE.md §1.3 要防的
     "未测当通过"模式，改成硬 `FileNotFoundError`，要跳过必须显式
     `validate_gold=False`（DD-0008）。

怎么解决的：见上。

测出什么数字：无（本轮不产生性能/准确率数字；BIRD 真实 425 条 gold
执行失败的数字要等真机联网下载 BIRD 全量后用
`validate_gold_sql` 实测才能拿到，目前只在 fixture 规模验证了机制本身）。
```

```
D0 续续续（同一 session）
做了什么：
  - A3：compare/ —— result_types.py（EQUAL/NOT_EQUAL/UNDECIDABLE 三态）、
    config.py（七维度显式配置：column_match_mode/row_order/
    null_equals_null/float_rel_tol+abs_tol/both_empty_is_equal/
    duplicate_row_mode/type_mode）、comparator.py（核心比对逻辑，
    出错或截断永不返回 EQUAL）、official_baselines.py（BIRD 官方
    `evaluation.py` execute_sql 的忠实复刻，真实抓取源码核对，见
    DD-0011）、golden_generator.py（生成 158 条合成对 + 3 条真实
    跨方言执行对 = 161 条，expected 字段全部留空）、consistency.py
    （对照我们的比对器与 BIRD 官方复刻，产出一致率与逐条分歧）。
    给 check_no_cheating.py 加了窄口径 allowlist 机制（DD-0010）。
    48 个单元/元测试全绿。真实生成了 `tests/golden/comparator_pairs.jsonl`
    （161 行，可以直接交给人工裁定 expected 字段）。

遇到什么问题：
  1. `official_baselines.py` 里故意复刻 BIRD 官方的
     `except Exception: return 0`（连同它的"任何异常都算错"这个已知缺陷）
     被自己的 check_no_cheating.py 命中——不是绕过检查，是给检查器加了
     一个要求写明理由、按规则名精确匹配的行内豁免机制（DD-0010），
     两个新发现的规则误报（`_example_rows` 的调试用途截断、
     official_baselines.py 缺 timeout）该改代码的改代码
     （补 timeout、把截断显式写进返回的 data 里），该加豁免的加豁免，
     不是笼统地把整个文件排除扫描。
  2. `type_mode="strict"` 一开始没有真的生效——`_values_equal`
     里数值容差分支写在 type_mode 判断之前，导致 strict 模式下
     `1 == 1.0` 照样判 EQUAL；无序比较路径的 `_canonicalize_value`
     则是另一个独立 bug：Python 自己的 `1 == 1.0`（且两者 hash 相等）
     让基于 Counter/set 的比较天然把 int 和 float 揉一起，跟 config
     完全无关。两个 bug 都是被
     `test_strict_type_mode_distinguishes_int_float_string` 这条测试
     真实跑挂之后才发现的——修法是给 strict 模式的规范化值加类型标签
     （`("float", v)` / `(type(v).__name__, v)`），让它们在 hash 层面
     就不可能撞上。
  3. 生成真实跨方言对抗样本时，PostgreSQL 的整数除法 `1/2` 通过 psycopg
     真实返回的是 `decimal.Decimal`，不是 float——`json.dumps(default=list)`
     无法序列化 Decimal（TypeError: 不可迭代），直接崩溃。这不是编出来的
     边界情况，是真实跑本地 PostgreSQL 撞出来的。改成显式识别
     `decimal.Decimal` 转成字符串（保留精确文本值，不是有损的 float 转换）。

怎么解决的：见上。

测出什么数字：无（本轮不产生性能数字）。真实产物：
`tests/golden/comparator_pairs.jsonl`（161 条候选对，expected 待人工裁定）。

面试追问三层（A3，答案已有真实依据，非空想）：
  - 为什么不直接用官方脚本当 GRPO 奖励函数？
    → 见 official_baselines.py 与 DD-0011：官方脚本 `set()` 比较无浮点
      容差、折叠重复行、不查行序、任何异常统一记 0。拿它当奖励函数，
      模型会学到"结果集去重"和"浮点数不需要精确"这类噪声信号。
  - "错判成对"和"对判成错"，哪个对训练更致命？
    → 错判成对（应该 NOT_EQUAL 却判 EQUAL）更致命：它会让模型在真正
      答错的样本上拿到满分奖励，且这类错误在训练曲线上完全不可见——
      这正是 CLAUDE.md 反作弊条款要防的"看起来正常的假数据"。
      "对判成错"至少会在验证集准确率上体现为一个可观察的下降。
      这也是本比对器"出错/截断永远不返回 EQUAL"这条硬规则的直接依据。
  - 空对空算相等吗？大量空对空会怎样？
    → 默认算相等（`both_empty_is_equal=True`），但这是一个可关闭的显式
      配置项，不是隐藏假设。大量空对空掩盖了模型"实际没有能力生成有效
      查询、只是恰好命中空结果"的可能——这个占比应该被 A4 的评估报告
      单独统计，而不是被平均分掩盖。
```

```
A4（同一 session）
做了什么：
  - eval/model_client.py —— `ModelClient` Protocol + `GenerationResult`；
    `HttpModelClient` 是真实的 OpenAI/vLLM 兼容 `/v1/completions` 客户端
    （本沙箱无 GPU、无可连的模型服务，标 `requires_network`，未在本沙箱
    实测，见 DD-0003）。`finish_reason == "length"` 是 CLAUDE.md §2.4
    OUTPUT_TRUNCATED 判据的来源，在 model_client 层面就已经决定，
    不是 eval 下游猜的。
  - eval/gold_cache.py —— gold SQL 执行结果按
    `(db_id, gold_sql_hash, db_file_hash)` 缓存（CLAUDE.md §7.1），
    真实落盘到 `artifacts/cache/gold_exec/`，`db_file_hash` 是整个
    db 文件内容的 sha256，文件一变哈希就变，不会踩着一个过期结果不放。
  - eval/records.py —— `PredictionRecord`，`counts_toward_denominator`
    是全项目准确率分母最终收口的地方：OUTPUT_TRUNCATED 和任何
    UNDECIDABLE（含 HARNESS_*）都被排除在分子分母之外，SYNTAX/SEMANTIC/
    TIMEOUT 算零分尝试但仍计入分母。
  - eval/runner.py —— `run_one_sample`/`run_eval`，真实的
    生成→执行→比对→落盘链路，`predictions.jsonl` 支持 `--resume-from`
    式续跑（已记录的 `sample_id` 直接跳过），每条新记录都原子地重写
    整个文件（`docs` 里写明了这是"eval 规模可接受、训练规模不可接受"
    的显式取舍，不是没意识到 O(n) 重写的代价）。
  - eval/metrics.py —— `compute_metrics`，HARNESS_* 占比 > 1% 直接
    `raise HarnessErrorFloodError`，拒绝产出任何报告（CLAUDE.md §1.3
    的"未测不算过"精神的另一种体现：系统性故障不该被平均成一个
    看起来合理的低分）；`execution_accuracy` 分母为 0 时是 `None`，
    不是 `0.0`。
  - eval/report.py —— `render_report_markdown`/`write_report`，
    两道硬门槛：manifest 被污染（`git_dirty`/`contaminated`/
    `degraded`）直接复用 `common/run_manifest.assert_not_polluted`
    拒绝；`REQUIRED_MANIFEST_FIELDS` 里任何一个字段是 `None`
    直接 `ReportError` 并把缺失字段名全部列出来（不是只报第一个）。
    报告正文第一行就显式印 tier，"denominator basis" 一节把
    total/excluded/denominator 三个数字都摊开写，不让读者只看一个
    孤零零的百分比。
  - `common/errors.py` 新增 `ReportError` 类和
    `REPORT_REQUIRED_FIELD_MISSING` 错误码。
  - 单元测试 51 条（tests/unit/a4/）、元测试 6 条（tests/meta/a4/，
    覆盖 CLAUDE.md §1.5 点名的 `test_eval_fails_on_exec_error_flood`
    和 `test_report_rejects_missing_metric` 两条）、烟测 1 条
    （tests/smoke/a4/，全链路跑通但规模缩到 4 条样本）。
    `make verify-a4` 58 个用例全绿。

遇到什么问题：
  1. `GenerationResult` 最初继承裸 `pydantic.BaseModel` 而不是项目统一的
     `ModelHubBaseConfig`，被
     `test_generation_result_forbids_unknown_field` 真实跑挂发现——
     裸 BaseModel 默认静默吞掉多余字段，正是 CLAUDE.md §4
     "拼错的 key 必须报错"要防的问题（DD-0013）。
  2. `run_eval` 最初的签名里有一个 `eval_tier` 参数，写完发现函数体
     完全没用它——"接受了却不用"的坏味道，删掉了，tier 的归属改为
     `RunManifest.eval_tier` + `report.py` 的硬校验（DD-0012）。
  3. `gold_cache.py::_read` 反序列化时把 `payload["code"]` 当裸字符串
     塞进 `ExecOutcome(code=...)`，没有转回 `ErrorCode` 枚举——虽然
     `ErrorCode` 是 StrEnum，`==` 比较不受影响，但项目里大量代码用
     `outcome.code is ErrorCode.SYNTAX` 做身份比较，反序列化出来的裸
     字符串 `is` 永远为 False。写测试前review代码时自己发现的，
     改成 `ErrorCode(payload["code"])`。
  4. 为了证明"预测 SQL 执行失败时绝不会再去跑 gold SQL / 绝不会执行
     被截断的 SQL"这类"跳过下游步骤"的行为是真实发生的，而不是相信
     代码读起来像是这样，测试里用了真实函数的计数 spy（wrap 真实
     `execute_isolated`/`get_or_execute`，照常真实执行，只是多计一次
     调用次数）而不是替换成假实现——运行时验证代码路径，而不是
     静态读代码猜代码路径。

怎么解决的：见上。

测出什么数字：无。A4 本身不产生任何性能/准确率数字（那是 A5 起
serve/ 真实起服务之后的事）；本轮所有"数字"都是测试断言里的小规模
构造值（4 样本/100 样本这类烟测/元测试规模),不满足 manifest 三个
清洁位、不构成可对外引用的数字。
```

```
A5（同一 session）
做了什么：
  - serve/schema_format.py —— 真实 `sqlite3` PRAGMA 反射（`PRAGMA
    table_info`/`PRAGMA foreign_key_list`），不是从 BIRD/Spider 的
    `tables.json` 元数据抄一份 schema——后者可能跟真实 db 文件本身对不上。
    两种渲染风格：`ddl`（CREATE TABLE 块，信息密度最高）、`compact`
    （`table(col1,col2)` 单行，给 A8 的 prompt 长度扫描用）。
  - serve/prompt.py —— `build_prompt`/`make_prompt_builder`，直接补上
    A4/DD-0012 里刻意留空的 `run_eval(prompt_builder=...)` 插槽；
    `SchemaCache` 按 db_id 缓存渲染结果，避免同一个 db 被评估的每条样本
    都重新反射一次 schema。
  - serve/kernel_status.py —— 把仓库里已经写好的 `preflight.py` G1/G2
    （causal_conv1d/fla 的真实 CUDA kernel 调用，不是只 import）重构成
    `serve/` 和未来 A9 门禁都能复用的函数，共享同一份检测逻辑而不是
    各写一遍。SKIP（未装/无 CUDA）和 FAIL（装了但跑不通/产出非法值）都
    折进 `degraded=True`，只有真正跑通的 PASS 才算数（CLAUDE.md §1.3
    白名单原则）。
  - serve/model_profile.py + configs/serve/model_profiles/{arctic_7b,
    qwen3_5_9b}.yaml —— KV/token 公式代码化，两份 YAML 配置的层数/KV头/
    head_dim 精确复现 FACTS.md §二表格，公式算出来的 56 KiB / 32 KiB
    跟文档值逐位对上（测试里独立算一遍，不是照抄实现）。weight_bytes
    和 Qwen3.5 的 gdn_fixed_state_bytes 在 YAML 注释里明确标成估算值，
    等真实下载/服务后用 vllm_log_parser.py 解析出的真实数字替换。
  - serve/vllm_log_parser.py —— 解析 vLLM 启动日志文本抽取 `# GPU
    blocks`/`Maximum concurrency`/`Loading model weights took` 三类
    确信度较高的已知稳定日志行；GDN 状态占用那一行的格式明确标成
    "未经真实日志核实的最佳猜测"，不是包装成看起来一样可信的数字
    （CLAUDE.md §12）。
  - `pyproject.toml` 给 `torch`/`transformers`/`peft`/`accelerate`/
    `datasets`/`vllm`/`causal_conv1d`/`fla` 加了 mypy
    `ignore_missing_imports` override——这些是 `train` extra 的重型可选
    依赖，本沙箱没装（无 GPU），运行时早已有真实 `try/except ImportError`
    兜底，这条 override 只是让类型检查器别为"没装的包没 stub"这件事
    报错，不改变真实机器上（这些包确实装了）的行为。
  - 单元测试 33 条（tests/unit/a5/）、元测试 2 条（tests/meta/a5/，
    证明"未验证的 GDN kernel 状态必须污染 manifest"这条链路真的能红——
    而且本沙箱确实没装 causal_conv1d/fla/torch，走的是真实的负向路径，
    不是构造出来的假失败）、烟测 1 条。`make verify-a5` 36 个用例全绿。

遇到什么问题：
  1. A5 没有在 CLAUDECODEPROMPTS.md 原文可查（那份文件是本 session
     一开始的上传附件，不在仓库里，context 压缩后原文不可再取）——
     本轮范围是从仓库里现有的三份文件（FACTS.md 的 KV 经济学表格、
     MODEL-SELECTION-FINAL.md §五"CLAUDE-CODE-PROMPTS.md 的改动"表格
     对 A5 的具体修订、preflight.py 已经写好的 G1-G5）交叉确定的，
     不是凭空猜的（DD-0014 里写了完整推理链）。
  2. 一开始把"KV 预算 → 最大并发数"的完整容量规划公式也想放进
     `model_profile.py`——写到一半意识到 MODEL-SELECTION-FINAL.md
     明确把"压测扫描...产出交叉曲线"划给了 A8，A5 的修订条目只提到
     "解析 KV block 数"，没提并发估算。把范围收窄成
     `kv_bytes_per_token` + `gpu_blocks_to_cacheable_tokens`
     两个直接服务于"log 解析出的数字该怎么解读"的函数，完整的容量
     规划公式留给 A8，没有在这一轮抢先做一半、留一半给 A8 时又要
     重新决定"这部分到底算谁的"。
  3. mypy 在激活 sqlexec 相关 extras 装了 duckdb/psycopg 的 venv 下
     对 `import torch` 报 import-not-found——这是本项目第一次在 src/
     里出现"根本没装、且连一次都没装过"的可选依赖（duckdb/psycopg
     从 A2 起就已经通过 sqlexec extra 装好了，torch 从来没装，因为
     `make install` 装的 extras 列表里没有 `train`）。没有给每处
     `import torch` 单独加 `# type: ignore`，而是在 `pyproject.toml`
     加了一条按模块名生效的 mypy override，B1/B2/B4/B5 训练那几轮会
     大量 `import torch`，一次性在配置层解决比每处散落注释更不容易漏。

怎么解决的：见上。

测出什么数字：无。`kv_bytes_per_token` 的 56 KiB/32 KiB 是从 YAML
里写死的架构事实（层数/KV头/head_dim，来自 FACTS.md 的已推算表格）
算出来的，不是实测；`weight_bytes`/`gdn_fixed_state_bytes` 在 YAML
注释里明确标了"估算，等真实运行替换"。
```

```
A6（同一 session）
做了什么：
  - gateway/auth.py —— API key 以 salted SHA-256 哈希存储/比对
    （`hmac.compare_digest` 常数时间比较），配置里只有哈希，没有明文密钥。
  - gateway/rate_limit.py + quota.py —— Redis 后端，真实本地 Redis
    （沙箱里 `redis-server` 已装但没启动，本轮启动后 `redis-cli ping`
    真实返回 PONG）。计数递增+首次设过期时间写成单条 Lua `EVAL`
    脚本，不是两次独立往返（DD-0015，含一处诚实修正——见"遇到什么问题"）。
    超限一律硬拒绝（`GatewayError`），不是打个警告继续放行。
  - gateway/routing.py —— schema token 长度阈值路由，`estimate_token_count`
    是标注清楚的粗略估算（~4 字符/token），`RoutingConfig.threshold_source`
    显式区分"估算"和"A8 实测"，防止一个拍脑袋的数字被悄悄当成
    生产级数字用（DD-0016）。
  - gateway/circuit_breaker.py —— 真实三态机（CLOSED/OPEN/HALF_OPEN），
    进程内状态，时钟可注入（测试不用真 sleep）。HALF_OPEN 只放行一个
    试探请求（`allow_request()` 会"claim"这个试探名额，第二次调用在
    结果落定前会被拒绝）——这是写测试时主动加固的一处，不是最初就想到的
    （见下）。
  - gateway/billing.py —— 单请求成本核算（prompt/completion 分开计价），
    `configs/gateway/billing.yaml` 里的价格明确标成"内部记账占位值"，
    不是真实市场价（本来就没有市场价可查——自建平台不是转售第三方 API）。
  - gateway/redis_support.py —— 唯一的 Redis client 工厂，强制显式
    socket timeout，避免每个模块各自记得/忘记设置。
  - `common/errors.py` 新增 `GatewayError` + 四个错误码
    （`AUTH_INVALID_API_KEY`/`RATE_LIMIT_EXCEEDED`/`QUOTA_EXCEEDED`/
    `CIRCUIT_BREAKER_OPEN`），复用 `Stage.SERVE`（不新增 Stage——
    CLAUDE.md §2.1 的 8 个 stage 是写死的项目宪法词表，`GATE` 专指
    A9 准入门禁，跟"网关"是两个概念，不能因为英文都叫 gate/gateway
    就混用）。
  - `tests/conftest.py` 新增 `TEST_REDIS_URL`/`requires_redis`/
    `redis_client` fixture，跟 A2 的 `requires_postgres` 同一个模式：
    对着真实本地服务测，服务不可达时优雅跳过而不是报错。
  - 单元测试 40 条（tests/unit/a6/）、元测试 3 条（tests/meta/a6/，
    50 线程真实并发压限流器/配额跟踪器）、烟测 1 条（auth→熔断→限流→
    路由→配额→计费全链路）。`make verify-a6` 44 个用例全绿。

遇到什么问题：
  1. `CircuitBreaker` 最初的 HALF_OPEN 实现里，`allow_request()`
     每次调用只看当前状态是不是 OPEN，HALF_OPEN 状态下会无限制放行——
     这违背熔断器"只放一个试探请求"的本意（真放行了一堆请求砸向
     还没恢复的后端，等于没有熔断）。写
     `test_half_open_allows_exactly_one_trial_request` 之前就自己发现了
     这个设计漏洞，加了 `_half_open_trial_claimed` 标志位，
     `allow_request()` 在 HALF_OPEN 下变成"消耗一次试探名额"而不是
     纯只读判断——这是写测试倒逼出的设计修正，不是测试之后才发现的
     bug。
  2. **本轮最重要的一次自我纠正**：写 DD-0015 时，最初的措辞声称
     "如果把 Lua 脚本换回两次往返，`test_rate_limit_and_quota_are_
     atomic_under_concurrency.py` 会因为竞态真的变红，已经手动验证过
     会偶发超额放行"——写完重新审视这句话时意识到这是编出来的：
     根本没有真的写一个两步版本去跑这个测试。而且仔细想了一下技术
     本质：Redis 的 `INCR`/`INCRBY` 单条命令本身就是原子的，跟后面
     是否紧跟一次独立的 `EXPIRE` 调用无关——两步版本的真实风险是
     "进程在 INCR 和 EXPIRE 之间崩溃，留下一个永不过期的键"，
     不是"并发下计数出错"。也就是说，这条并发测试大概率对两步版本
     一样会通过，它验证的是"限流器/配额跟踪器在真实并发下精确卡在
     配置上限"这件事本身，不是 Lua 脚本这个实现选择的回归测试。
     发现这处过度声称后，把 DD-0015 的"实测数字"段落和元测试文件的
     docstring 都重写了，去掉编造的"已验证"说法，把测试真正验证的
     内容和 Lua 脚本真正解决的问题（崩溃窗口导致键永不过期）分开
     说清楚，不再暗示两者是同一件事。CLAUDE.md §12"不确定就说
     不确定"不只约束对外的性能数字，也约束写给自己看的设计记录——
     这次是自己在收尾复核时抓到的，没有等用户指出来。

怎么解决的：见上。

测出什么数字：无。billing.yaml 里的价格、rate_limit/quota/
circuit_breaker 的阈值都在 YAML 注释里明确标成"占位运营值，等真实
流量再调"，routing.yaml 的阈值标成 `threshold_source: "estimate"`。
```

```
A7（同一 session）
做了什么：
  - monitor/mfu_mbu.py —— `compute_mfu`/`compute_mbu` 是 FACTS.md §四
    公式的直接代码化，`classify_bottleneck` 把"MBU 高+MFU 低=显存瓶颈"
    这类定性判断变成显式、配置驱动的阈值判定（DD-0017）。
  - monitor/hardware_bench.py —— 把 `preflight.py` E1/E2 的真实 BF16
    GEMM / 显存 D2D copy 微基准重构成可复用函数（和 A5 把 G1/G2 重构成
    `serve/kernel_status.py` 是同一个模式），测不出时返回 `SKIP` +
    `None`，不产出未校准的数字。
  - monitor/gpu_metrics.py —— 把 preflight D3 的 DCGM PROF 字段探测
    （`dcgmi dmon`）和 nvidia-smi 查询重构成可复用函数；`NvidiaSmiSnapshot`
    的字段刻意命名成 `*_time_occupancy_pct` 而不是 `*_utilization_pct`，
    从命名上就防止这个数字被下游误当成工作量度量转述（FACTS.md §四：
    `DCGM_FI_DEV_GPU_UTIL` 的官方定义是"过去采样周期内至少有一个 kernel
    在执行的时间占比"，不是工作量）。
  - monitor/cache_metrics.py —— `PrefixCacheStats`，命中率为 0 请求时是
    `None` 不是假装成 `0.0`；这也是 DD-0001 选 vLLM 而非 SGLang 时
    "固定 schema 前缀命中率高"这个论据第一次有了可以真实测量的落点。
  - monitor/exporter.py —— `GatewayMetrics`，真实 `prometheus_client`
    `CollectorRegistry`（不用全局默认 registry，因为同进程多实例/
    并行测试会撞名），覆盖请求计数/延迟直方图、限流/配额拒绝计数、
    熔断器状态（复用 A6 的 `CircuitState`）、MFU/MBU/prefix-cache
    命中率仪表盘。PLAN.md"FastAPI 网关 + Redis + Prometheus/Grafana"
    这条平台栈里 Prometheus 的部分。
  - `configs/monitor/bottleneck_thresholds.yaml` —— 阈值明确标注"未经
    真实 MFU/MBU 分布校准的占位值"，跟 A6 的运营参数配置同一套诚实标注
    习惯。
  - 单元测试 36 条（tests/unit/a7/）、元测试 2 条（tests/meta/a7/，
    验证"测不出的硬件峰值"真的会在 manifest 上保持 `None`，不会被
    悄悄替换成规格值）、烟测 1 条。`make verify-a7` 39 个用例全绿。

遇到什么问题：
  1. **本轮发现的一个跨轮次、影响全项目的真实 bug**：
     `tests/unit/a7/` 里最初有一个 `test_configs_load.py`，和 A6 已经
     存在的 `tests/unit/a6/test_configs_load.py` 重名。单独跑
     `pytest tests/unit/a7` 完全正常，但跑全量 `pytest`（没有路径过滤）
     时会报 `import file mismatch`——原因是 `tests/unit/a6/` 和
     `tests/unit/a7/` 都没有 `__init__.py`（从 A0 起就是这个约定，
     pytest 把这类目录当作顶层模块名的插入点），两个同名文件被当成
     同一个顶层模块 `test_configs_load` 冲突。这是每轮单独跑
     `make verify-aX` 测试永远不会暴露、只有跑全量回归测试才会撞见的
     一类 bug——之前 A0-A6 因为没有出现过重名文件而侥幸没触发。
     把 A7 的文件改名成 `test_monitor_configs_load.py` 解决了这一次，
     但这是个会在后续任何一轮再次发生的结构性风险（只要两个不同轮次
     恰好取了同一个测试文件名）。已经用
     `find unit meta smoke -name "test_*.py" | xargs -n1 basename | sort | uniq -d`
     确认目前项目里没有其它重名，后续每轮收尾前会用同一条命令复查。
  2. 本沙箱没有 `nvidia-smi`/`dcgmi`（连 GPU 驱动栈都没装，比 A5/A7
     "有 CUDA 但没装 fla/causal_conv1d"更进一步的"完全没有 GPU 相关
     工具链"），`gpu_metrics.py` 的测试走的是真实的"命令不存在"分支
     （`shutil.which` 真的返回 `None`），不是构造出来的。
  3. 排查上面第 1 条时，顺手发现本地工作区里多了一个没被 git 追踪的
     `tests/artifacts/data/...` 目录——查下来是 A1 的
     `gold_validation.py::write_gold_exec_failures` 和
     `pipeline.py::write_data_build_report` 一直硬编码相对路径
     `Path("artifacts/data")`，没有像 `eval/report.py::write_report`/
     `common/run_manifest.py::write_manifest` 那样开放
     `artifacts_root` 参数——测试没显式指定输出目录时，产物就写到了
     "当时进程 cwd 恰好是哪"，本该被 `.gitignore` 的 `artifacts/*`
     挡住，但 cwd 一旦不是仓库根目录（比如曾经从 `tests/` 目录下跑过
     一次 pytest），产物就漏到了 `tests/artifacts/`（`.gitignore` 的
     `artifacts/*` 是相对仓库根目录的模式，不匹配 `tests/artifacts/*`）。
     这是本项目至今唯一一处"产物路径依赖 cwd 而不是显式参数"的地方，
     顺手补上了 `artifacts_root: Path = Path("artifacts/data")`
     参数（跟 `write_report`/`write_manifest` 同一个默认值+可覆盖模式），
     三个受影响的 A1 测试文件改成显式传 `tmp_path` 下的临时目录，
     不再依赖进程 cwd；已删除误产生的 `tests/artifacts/` 和
     `artifacts/data/`（后者本就在 .gitignore 里，删除只是清理本地
     工作区，不影响仓库状态）。这不是 A7 的功能范围，但是在为 A7 收尾
     确认 `git status` 时顺带抓到并修掉的真实一致性问题，没有留到
     "以后再说"。

怎么解决的：见上。

测出什么数字：无。本沙箱无 GPU，`hardware_bench.py` 的两个测量函数
在这里只能验证"真实测不出时诚实返回 None"这一条路径；真实 TFLOPS/
带宽数字要等真机验证（DD-0003）。
```

```
A8（同一 session）
做了什么：
  - bench/gpu_guard.py —— `check_gpu_exclusivity`/`guard_gpu_exclusivity`，
    真实 `nvidia-smi --query-compute-apps` 进程数检测（CLAUDE.md §5.2
    "压测机独占"）。本沙箱没有 nvidia-smi，测的是真实 SKIP 路径；
    `guard_gpu_exclusivity` 对 SKIP/FAIL 一视同仁拒绝启动（白名单原则：
    "测不了"不等于"干净"）。
  - bench/load_test.py —— `run_load_test`，`ThreadPoolExecutor` 并发
    压测一个 `ModelClient`（复用 A4 的 Protocol，不重新定义接口），
    最近邻百分位（P50/P95/P99）纯 Python 实现，失败请求按异常类型分类
    计数（`error_breakdown`），不静默吞掉。
  - bench/capacity_planning.py —— `estimate_max_concurrent_requests`/
    `find_concurrency_crossover_point`，A5/DD-0014 里明确留给 A8 的
    容量规划公式。**过程中发现并修复一个真实的公式语义 bug**——GDN
    固定态开销该算"每并发请求"还是"全卡一次性"，只有前者能复现
    MODEL-SELECTION-FINAL.md 写的"1k–3k 之间交叉"结论（DD-0018，
    全过程详细记录）。
  - bench/cost_model.py —— `cost_per_million_tokens_usd`/
    `estimate_request_cost_usd`，GPU 每小时租金 ÷ 实测 tok/s 反推
    $/请求，`gpu_hourly_cost_usd` 强制显式传参、代码里没有任何默认值——
    FACTS.md 只公开过训练用 2×A100 的租金区间，没有本项目实际用的
    RTX 5090 推理卡的确认租金，不能凭空补一个"看起来合理"的数字。
  - bench/sweep.py —— `run_sweep`，二维 sweep（并发 × prompt 长度）
    编排，启动前强制走 `guard_gpu_exclusivity`（`skip_gpu_guard` 仅供
    测试用，绝不能在真实压测里打开）。
  - `configs/bench/capacity_planning.yaml` —— GPU 显存/利用率/运行时
    开销直接照抄 FACTS.md 自己写的公式（32GiB × 0.92 − 2.5GiB），
    不是另编一套数字。`configs/bench/sweep_grid.yaml` 是压测网格设计
    参数（不是测量值）。
  - 单元测试 33 条（tests/unit/a8/）、元测试 4 条（tests/meta/a8/，
    其中一条把 DD-0018 发现的错误公式原样留作对照实现，另一条证明
    GPU 独占门禁真的能挡住一次端到端的压测启动）、烟测 1 条。
    `make verify-a8` 38 个用例全绿。

遇到什么问题：
  1. **本轮最重要的发现**：见 DD-0018——GDN 固定态开销的语义 bug。
     这不是"写测试时顺手改了改代码让测试过"，而是先写出一个明确、
     独立于代码实现的验证目标（"真实配置应该复现文档写的交叉行为"），
     拿这个目标去检验代码，代码没通过，回头去查为什么、找到真实原因
     （循环状态是逐请求的，不是逐卡的）、修正语义、重新验证——这正是
     CLAUDE.md 反复强调的"先写会真的变红的检查，再让实现去满足它"，
     不是反过来拿实现结果去调检查的阈值。
  2. `bench/cost_model.py` 差点想给 `gpu_hourly_cost_usd` 一个"看起来
     合理"的默认值（比如照抄训练那台 2×A100 算出来的单卡时租金），
     写文档字符串时意识到这是在编数字——FACTS.md 明确只对训练 GPU
     报过区间，没有针对实际使用的 RTX 5090 推理卡的确认价格，
     两种卡的市场行情不能互相替代。改成强制必填参数，不留任何隐式
     兜底。

怎么解决的：见上。

测出什么数字：无。A8 本身不产生真实压测数字（本沙箱无 GPU，无法真的
起 vLLM 服务测 QPS/P99）；容量规划的"交叉点在 1k–3k 之间"是复现文档
既有结论的正确性验证，不是新测出的数字；成本模型的函数本身没有任何
硬编码价格，等真实压测拿到 tok/s 之后才能算出真实数字。
```

```
A9（同一 session）
做了什么：
  - gate/types.py —— `GateDecision`（PASS/REJECT/NOT_APPLICABLE 三态，
    不是裸 bool——"没有基线可比较回归"跟"比较过、确认没有回归"是两码事，
    NOT_APPLICABLE 显式区分开）、`GateResult`/`GateVerdict`。
  - gate/accuracy_gate.py（闸1）—— 绝对准确率下限，`execution_accuracy
    is None`（分母为0）直接 REJECT，不当成"没数据所以先放过"。
  - gate/regression_gate.py（闸2，CLAUDE.md §1.5 点名要求的回归检测器）
    —— `find_regressions` 找出"基线答对、候选答错"的 sample_id 集合，
    只在基线和候选都有的样本上比较；`check_regression_gate` 在没有
    基线时返回 NOT_APPLICABLE（仅限第一个上线模型），不是默默当 PASS。
  - gate/safety_gate.py（闸3）—— 直接对 270 条 CRUD 对抗样本的真实
    gold_sql 跑一遍 sqlexec，验证平台只读保证本身（不是候选模型的
    行为），见 DD-0020。
  - gate/pollution_gate.py（闸4）—— 复用 A0 的 `RunManifest.is_polluted`。
  - gate/truncation_gate.py（闸5）—— 输出截断率上限，呼应
    MODEL-SELECTION-FINAL.md 陷阱2（量化+thinking 模式截断率飙升）。
  - gate/admission.py —— 五道闸全部跑完不短路（DD-0019），
    `AdmissionGateConfig` 汇总五份子配置。
  - release/registry.py —— `ModelRegistry`，文件系统落盘
    （`artifacts/registry/<model_id>/<version>.json`，原子写），
    状态机 CANDIDATE→APPROVED/REJECTED→DEPLOYED→ROLLED_BACK，
    每条记录带完整 `GateVerdict` 历史（不是只存最终 pass/fail 一个
    bool）。非法状态迁移（比如没通过闸就想标 DEPLOYED）显式 `ValueError`。
  - `configs/gate/admission.yaml` —— 除 `safety.max_allowed_unblocked: 0`
    （这个不是占位值，是 A1 文档原话"270/270"的直接体现）外，其余阈值
    明确标注"占位运营值，等真实 baseline 模型的 quick-eval 数字出来
    再调"。
  - 单元测试 36 条（tests/unit/a9/）、元测试 6 条（tests/meta/a9/，
    覆盖 CLAUDE.md §1.5 点名的两个必需项 `test_gate_can_fail` /
    `test_regression_detector_fires`，外加一条证明安全闸的 REJECT
    分支本身可达、不是永远进不去的死代码）、烟测 1 条（门禁 + 注册表
    全链路）。`make verify-a9` 43 个用例全绿。

遇到什么问题：
  1. "五道闸"具体是哪五道、"ckpt-A/B/D 分别被哪道拦下"的原始提示词
     文本已经不在上下文里可查——从 PLAN.md/FACTS.md/
     MODEL-SELECTION-FINAL.md 交叉印证，加上 Round 0 就写好的
     `gate/__init__.py` 骨架 docstring（"five checks"）反推出这五道闸
     的具体内容，并在 DD-0019/DD-0020 里写清楚推理依据，不是拍脑袋编的。
  2. 安全闸最初设计成"跑模型生成 SQL，再看有没有被拦下"，写文档字符串
     时意识到这个设计在系统保证下没有信息量（sqlexec 的只读连接让写
     操作在机制上永远不可能成功，不管模型输出什么）——改成直接测
     沙箱自身的只读保证（DD-0020），并专门写了一条元测试
     （`test_safety_gate_can_fail.py`）证明这道闸的 REJECT 分支
     不是永远进不去的死代码：用一个刻意"标记成对抗样本但其实是普通
     SELECT"的样本，模拟"沙箱保证已经悄悄失效"这种情况，验证闸真的
     会在这种情况下报 REJECT。

怎么解决的：见上。

测出什么数字：无。本沙箱没有真实候选模型/真实 baseline 模型可以跑出
真实的 accuracy/regression/truncation 数字；`configs/gate/admission.yaml`
里除了 safety 阈值都标注了"占位待真实数据校准"。
```

```
A10（同一 session）
做了什么：
  - release/canary.py —— `CanaryConfig`（阶梯必须递增且以 100 收尾，
    `model_post_init` 里显式校验，不留 `[5,25,50]` 这种漏了最后一档
    的配置能悄悄通过的空子）、`route_canary_traffic`（`hashlib.sha256
    (request_key) % 100 < canary_percentage`——同一个 request_key 永远
    落在同一个臂上，不是 `random.random()`；重试请求/同一用户会话
    不该在 stable/canary 之间跳变，这是灰度路由的真实要求，副作用是
    测试里也不用去 seed 一个全局 RNG）、`CanaryRolloutState`（frozen
    dataclass，`stage_index`/`requests_at_current_stage`/`status`）、
    `start_rollout`/`record_canary_request`/`try_advance_stage`/
    `mark_rolled_back` 四个纯函数，不带副作用，调用方自己负责用返回值
    替换旧状态。
  - release/rollback.py —— `CanaryHealthSample`（真实 sqlexec 分类事实：
    succeeded/is_harness_error/latency_s，不是一个人工健康分）、
    `RollbackConfig`、`summarize_canary_health`（复用 A8 的
    nearest-rank 百分位算法算 p99 延迟）、`evaluate_canary_health`——
    复用 A9 的 `GateDecision`/`GateResult` 三态而不是另起一个布尔
    'should_rollback'：REJECT=该回滚，PASS=健康，NOT_APPLICABLE=样本
    量不够、还不能判断（CLAUDE.md §1.3：没测够不能悄悄读成"健康"）、
    `trigger_rollback` 调 A9 `ModelRegistry.mark_rolled_back` 真的把
    注册表状态翻过去，不是只返回一个决策对象扔给调用方自己处理。
  - release/online_sampling.py —— `sample_recent_predictions`（seeded
    `random.Random(seed)`，CLAUDE.md §7"采样类评估必须记录 seed"）、
    `predictions_to_health_samples`（`succeeded = exec_code is
    EXEC_OK`，见 DD-0022）。
  - release/orchestrator.py —— `run_canary_cycle` 把抽样→健康评估→
    回滚/推进串起来：REJECT 触发真实回滚并把 rollout 状态标
    ROLLED_BACK，NOT_APPLICABLE 原地不动等更多数据，PASS 才尝试推进
    一档（推进到最后一档本身不算完成，见 DD-0021，需要在 100% 这一档
    再跑一轮健康检查才 COMPLETED）。
  - `configs/release/canary.yaml`（阶梯 `[5,25,50,100]`，第一档 5% 是
    PLAN.md 原话，其余是合理运营默认值，非实测）、
    `configs/release/rollback.yaml`（错误率/系统错误率/p99 延迟三个
    阈值，同样标注"占位运营值"）。
  - 单元测试 46 条（tests/unit/a10/：canary 状态机 19 条、rollback
    健康评估 13 条、online_sampling 10 条、orchestrator 6 条，外加
    共享 `fakes.py` 三个 PredictionRecord 构造器：healthy/failing
    （模型侧 SYNTAX）/harness_error，让"模型错"和"系统错"两类信号在
    测试里能分开构造）、元测试 2 条（tests/meta/a10/
    test_canary_rollback_fires.py，见下）、烟测 1 条（完整两阶段
    rollout：5%→100%→COMPLETED，注册表状态全程保持 DEPLOYED 不误伤）。
    `make verify-a10` 49 个用例全绿；`mypy --strict` 对
    `src/modelhub` 全量 80 个源文件干净；全项目 497 个测试
    （不含 requires_gpu/requires_network）全绿，没有引入跨轮 regressions。

遇到什么问题：
  1. 写 `test_stage_out_of_range_rejected` 时想用 `[5, 150, 100]` 去测
    "中间某一档超过 100"，结果发现这个用例根本走不到范围校验那一步——
    `CanaryConfig.model_post_init` 先查 ascending，`[5,150,100]` 排序后
    是 `[5,100,150]`，跟原序列不一致，直接在"必须递增"这一步就被拒了。
    往回想了一下：只要"递增"和"末尾必须是 100"两条都成立，中间任何
    一档就自动被夹在 `(0, 100]` 里，不可能单独超过 100——真正能触发
    "超出范围"校验、又不提前撞上前两条校验的，只有非正数这一种情况。
    把测试改成 `[-5, 50, 100]`，测的是校验链条里真正可达的分支，不是
    凭直觉编一个"看起来应该触发"但实际到不了的用例。
  2. 写 `try_advance_stage`/`run_canary_cycle` 的完成语义测试时，最初
    假设"推进到 `stages` 里的最后一个值（100%）"和"标记为 COMPLETED"
    是同一次调用发生的事——测试跑了两个红：`test_canary.py::
    test_try_advance_stage_completes_at_last_stage` 和
    `test_orchestrator.py::test_healthy_traffic_at_last_stage_completes`。
    回去看实现代码：`try_advance_stage` 只有在"当前已经站在最后一档
    **并且**这一档也观测满了 `min_requests_per_stage`"才会返回
    COMPLETED——从上一档推进到 100% 的那次调用只是把 `stage_index`
    指向了最后一档，状态仍是 `IN_PROGRESS`。这是实现代码本来就对的
    设计（100% 阶段本身也需要被验证一遍才能收尾，见 DD-0021），是
    测试断言写错了，不是代码需要改——照实现的真实行为把两条测试改成
    两轮调用，不是反过来放松实现去迁就一开始想当然写的断言。
  3. session 因为上下文压缩重启后，shell 里的 venv 没有重新
    `source .venv/bin/activate`，第一次跑 `mypy src/modelhub` 用的是
    系统 Python，报了一堆 `pydantic`/`redis`/`duckdb` 等找不到模块的
    假错误——不是真的代码问题，是环境没激活。后续命令统一改用
    `.venv/bin/ruff`/`.venv/bin/mypy`/`.venv/bin/pytest` 显式路径，
    不依赖 shell 状态跨 Bash 调用持久化（工具本身就说了 shell state
    不持久化，只有 cwd 持久化）。

怎么解决的：见上。

测出什么数字：无。A10 不产生真实灰度流量或回滚事件——本沙箱没有真实
线上服务、没有真实模型部署；`configs/release/canary.yaml` 和
`configs/release/rollback.yaml` 里的阶梯档位与阈值都标注了"占位运营
值，等真实生产流量出现后再调"，唯一不算占位的是第一档 5%（PLAN.md
原话）。
```

```
A11（同一 session）
做了什么：
  - bench/experiments/common.py —— `check_serving_precondition` 复用
    A0 的 `assert_not_polluted`（严格超集，覆盖 degraded/contaminated
    外加 git_dirty，见 DD-0023），`run_experiment_group_isolated` +
    `ExperimentGroupOutcome`——PLAN.md"六组实验...每组独立、失败不
    阻断、各自写 manifest"这句话对应的具体代码：捕获单组异常、记完整
    traceback 到日志（`log_exception`，CLAUDE.md §8），把失败包成
    显式 COMPLETED/FAILED 结果返回，不吞异常也不让一组的失败拖垮
    另外五组。
  - bench/experiments/vllm_metrics.py —— 真实 Prometheus 文本格式
    解析（`parse_prometheus_counter`），group 2/3 共用；缺失的
    counter 返回 `None`，不是编出一个 0。
  - 【1】quantization.py —— AWQ/GPTQ/FP8 三方（NVFP4 排除，见
    DD-0024）；`quantized_weight_bytes` 按 bit 宽度线性折算，喂给 A8
    `estimate_max_concurrent_requests` 算"用显存换并发"，不是"省
    显存%"；`quantize_model_checkpoint` 是三个真实量化库调用（AutoAWQ/
    GPTQModel/llm-compressor），本沙箱没装这些库也没有 GPU，走的是
    真实 ImportError→SKIP 分支（同 A5 kernel_status.py 的纪律）；
    GPTQ 分支额外在 SKIP 之后留了一个 `NotImplementedError`（校准集
    没有真实数据源可接，宁可硬失败也不编一个假校准集）；
    `run_quantization_accuracy_check` 直接复用 A4 `run_eval` + A9
    `run_admission_gate`，量化模型不是只看 perplexity；
    `run_truncation_cross_experiment` 是 v2 新增的 thinking×量化
    截断率对照，单元测试真的构造了一个 finish_reason="length" 的
    fake client 验证截断率能从 0 变成 1。
  - 【2】prefix_cache.py —— `build_interleaved_traffic` 实现"请求
    交错度"（PLAN.md 明确禁止"同一个db连发1000次刷命中率"的易骗
    做法），单测验证了 degree=1 时真的逐个切换 db、degree 够大时
    真的退化成"同一个db连发N次"；命中率强制来自 vLLM 真实 metrics
    （`measure_prefix_cache_hit_rate`，缺 counter 直接 raise，不
    补 0）；`ArchitectureCacheComparison`（v2 新增）只是把 Arctic/
    Qwen3.5 两组真实 `PrefixCacheStats` 并排放，不臆造折扣数字。
  - 【3】speculative_decoding.py —— MTP（Qwen3.5 原生头）vs n-gram
    （Arctic 无匹配 EAGLE 头）双路，`measure_concurrency_decay_curve`
    复用 A8 `run_load_test` 在每个并发档位各测一次 QPS，
    `SpeculativeDecayCurve.collapses_under_concurrency` 判断加速比
    是否跌到 ≤1——元测试用一个刻意变慢的 fake speculative client
    证明这个判断真的能触发，也用一个真的更快的 client 证明它不是
    永远红。
  - 【4】constrained_decoding.py —— 手写 SQLite SELECT 子集 EBNF
    语法（不含 DDL/DML/CTE/窗口函数，PLAN.md 明确说不写全量语法）；
    `compile_sqlite_select_grammar` 是本轮唯一一个只受 GPU 缺失
    间接影响、CPU 就能跑的真实调用（xgrammar 语法编译本身不需要
    GPU），本沙箱没装 xgrammar，走真实 SKIP；`syntax_legality_rate`
    专门取"非 SYNTAX"而不是复用 A4 `EvalMetrics.syntax_valid_rate`
    （后者是 EXEC_OK 口径，混了 SEMANTIC/TIMEOUT），因为约束解码
    只保证语法合法；`ConstrainedDecodingResult.execution_accuracy_
    change` 的文档字符串专门写清"诚实边界"：语法合法不等于语义正确，
    没做 schema 感知就照样能生成列名不存在的合法 SQL。
  - 【5】engine_comparison.py —— vLLM/SGLang 都是 OpenAI 兼容
    `/v1/completions`，直接复用 A4 `HttpModelClient` + A8
    `run_load_test`，没有新写引擎专属客户端；TensorRT-LLM 不参与
    对比，理由写进 DD-0025（SM120/121 上 FMHA cubin 缺失会回退
    unfused MHA，且只能走 NGC 容器安装，两条理由缺一都不足以排除，
    合在一起才是真正原因）。
  - 【6】mfu_mbu_comparison.py（v2 新增）—— 直接复用 A7
    `compute_mfu`/`compute_mbu`/`classify_bottleneck`，不重新发明
    公式；`ArchitectureMfuMbuComparison` 把"省显存容量"（KV/token，
    复用 A5 `kv_bytes_per_token`）和"省显存带宽"（MBU）显式拆成两个
    独立属性，因为 PLAN.md 原话"Qwen3.5 省的是显存容量，不是显存
    带宽"说的是两个不同的资源，混成一个数字会把这句话说岔。
  - configs/bench/experiments/ 六个配置文件，每个都写清哪些字段是
    FACTS.md/A8 已有的真实假设复用（GPU 显存预算、model_profiles 的
    weight_bytes）、哪些是本轮新增的运营占位值、哪些字段（如
    peak_bf16_flops/peak_bw_bytes_s）在真实运行前必须被
    `hardware_bench.py` 的真实测量结果替换掉——不是能跑就算数的
    随手数字。
  - pyproject.toml 新增 `experiments` 可选依赖组
    （autoawq/gptqmodel/llmcompressor/xgrammar）。
  - Makefile 新增 `verify-a11-1`..`verify-a11-6` 六个独立目标
    （PLAN.md"每组能单独 make verify-a11-<n>"的字面要求），显式目标
    优先于 `verify-%` 模式规则，泛用的 `make verify-a11` 仍然跑
    整个 tests/{unit,meta,smoke}/a11/。
  - 单元测试 68 条（tests/unit/a11/）、元测试 7 条（tests/meta/a11/，
    覆盖前置门槛的白名单纪律 + 推测解码收益衰减检测器能红两条主线）、
    烟测 1 条（两组真实跑通、一组故意失败，证明编排器的隔离在端到端
    场景下也成立）。`make verify-a11` 全绿；`make verify-a11-1`..
    `verify-a11-6` 逐个验证过；mypy --strict 对 90 个源文件干净；
    全项目 565 个测试全绿，无跨轮 regressions。

遇到什么问题：
  1. `tests/unit/a11/test_orchestrator.py` 与 A10 已有的
    `tests/unit/a10/test_orchestrator.py` 撞了 basename——这正是 A7
    踩过、当时加进了"每轮结束前跑一遍 basename 去重检查"这条纪律的
    同一类 bug，这次流程本身逮住了它，改名成
    `test_experiment_orchestrator.py` 后全项目测试套件恢复全绿。
  2. GPTQ 那条测试最初写成"预期抛 NotImplementedError"，实际跑起来
    才发现：本沙箱没装 gptqmodel，`quantize_model_checkpoint` 的
    ImportError 兜底分支会在到达 NotImplementedError 之前就先返回
    SKIP——`NotImplementedError`只有在 gptqmodel 真的装了、真的进到
    "校准集没有真实数据源"这一步时才会触发，这台沙箱测不到那一分支。
    照实现的真实分支顺序把测试改成断言 SKIP，并在测试里写清楚为什么
    测不到 NotImplementedError 分支，而不是反过来改实现去凑一个
    一开始想当然写的断言。
  3. session 因上下文压缩重启后 shell 里的 venv 又没激活（同 A10 遇到
    过的问题），第一次跑 `mypy src/modelhub` 又报了一堆假的
    import-not-found——继续沿用 A10 定下的规矩：所有校验命令一律走
    `.venv/bin/<tool>` 显式路径。

怎么解决的：见上。

测出什么数字：无。A11 六组实验没有一组产生可引用的真实数字——本沙箱
既无 GPU 也无真实 vLLM/SGLang 服务实例；量化的"用显存换并发"、命中率、
加速比、语法合法率提升、引擎吞吐对比、MFU/MBU 全部是真实公式/真实
编排代码等一次真机 GPU 跑通。`configs/bench/experiments/*.yaml` 里
除了少数直接复用 FACTS.md/A5/A8 已核实数字的字段外，其余全部标注
"占位/估算，等真实测量替换"。
```

```
A12（同一 session）
做了什么：
  - release/incident_log.py —— 补上 A9/A10 都写了要做、都没真正接线
    的 append-only `docs/incident-log.md` 写入器（DD-0026）：
    `gate_rejection_incident`/`canary_rollback_incident` 从真实
    `GateVerdict`/回滚原因构造记录，`append_incident` 走
    读-改-原子写整份文件替换的模式（同 eval/runner.py 的
    `_append_and_persist`），`count_incidents` 供交付检查表统计用。
    刻意不在 `run_admission_gate`/`trigger_rollback` 内部加这个 I/O
    副作用——保持这两个已交付、已测试的函数纯/薄副作用，让调用方
    显式记录。
  - gateway/app.py —— A6 写的六个模块（auth/rate_limit/quota/
    circuit_breaker/routing/billing）第一次被真正串成一个可运行的
    FastAPI 应用（DD-0027），顺序是鉴权→限流→路由→熔断→调用模型→
    配额→计费，模块文档字符串写清每一步为什么在这个位置。同时接上
    A7 的 `GatewayMetrics`（A6/A7 之间此前也没接过线）：每次请求记
    latency+outcome，熔断器状态变化实时更新 Gauge，新增 `/metrics`
    端点。配额检查放在生成之后（`quota.py`唯一入口需要本次请求的
    真实 token 数才能判断，生成前不存在这个数）——DD-0027 写清这不是
    绕开 PLAN.md"超额429"的字面意思，而是尊重 A6 已交付接口形状的
    诚实实现，真正的前置检查需要给 `quota.py` 新增只读接口，记进
    交付检查表的已知局限，不是这轮顺手改掉两轮前的模块。
  - scripts/render_docs.py —— docs/ 下的性能数字第一次由脚本从
    artifact/config 生成而不是手写（PLAN.md 明确要求）。三个页面，
    两种"真实"分开标注（DD-0028）：`capacity-plan.md`/
    `crossover-curve.md`是"PLANNING ESTIMATE"（公式真实，输入是文档
    承认的估算值——真跑出来 Arctic-7B/Qwen3.5-9B 在 2000 tokens 交叉，
    与 MODEL-SELECTION-FINAL.md 记载的 1k–3k 区间吻合），
    `metrics-summary.md`是"PENDING"（六个核心数字里五个既无公式也无
    真实 run，第六个即交叉点指向前两个页面）。`discover_runs`扫描
    `artifacts/runs/*/manifest.json`，复用 A0 `RunManifest.
    is_polluted`排除污染 run，不是重新发明一套判断。
  - scripts/audit_pollution.py —— 复用 `render_docs.discover_runs`
    的同一次扫描，输出可读报告 + 非零退出码（供 CI/`make verify-a12`
    挂钩），不重新走一遍 artifacts/runs 的目录遍历。
  - scripts/demo_stub_model_server.py + scripts/demo.py —— `make
    demo`的完整实现："起服务→发请求→展示监控→触发一次门禁拦截→
    展示incident-log"六步全部是真实代码路径：真实 FastAPI 网关应用
    （TestClient 驱动，真实 ASGI 请求生命周期）、真实本地 Redis、
    真实 sqlexec 执行 + compare 比对（对着真实 SQLite db）、真实
    Prometheus `/metrics` 抓取、真实 A9 五道闸 REJECT verdict（喂
    20/20 全错的候选）、真实追加 `docs/incident-log.md`。唯一不真实
    的一环是"模型"本身——`demo_stub_model_server.py`是一个在模块
    文档字符串里反复标"NOT A REAL MODEL"的固定响应桩，因为本沙箱
    没有 GPU；这个桩活在 `scripts/`（不是 `src/modelhub/`），呼应
    CLAUDE.md §1.4"mock 只能在 tests/"的精神——demo 专用的桩同样不
    该混进真实包。手动完整跑通两次（子进程干净启动+终止，无残留
    进程），docs/incident-log.md 里两条真实 GATE_REJECTION 记录就是
    这两次跑出来的，不是编的示例。
  - docs/architecture.md —— Mermaid 架构图，控制在能手画的复杂度
    （请求路径 6 节点 + 离线训练路径 3 节点 + 发布路径 5 节点），
    sqlexec/compare 因为是"四条链路共用"的横切模块、画出来会破坏
    "简单"这个约束，在正文里显式说明为什么没画而不是假装漏掉了。
  - docs/interview-qa.md —— 全部 19 轮的"面试追问"，逐字从
    CLAUDECODEPROMPTS.md 原文核对抽取（不是凭记忆转述），12 轮有真实
    问题列出清单+空复选框，7 轮（Round0/A0/A1/A4/A6/A12/B3）原始提示词
    本来就没有面试追问段落，单独一节列出来说明，不是假装漏了或编几个
    凑数。
  - docs/delivery-checklist.md —— PLAN.md A12 结尾要求的交付检查表，
    六个核心数字/交叉曲线/门禁拦截与回滚次数/决策记录条数/快评全评
    归属/诚实清单逐项回答，每个数字都标了怎么复现（grep 命令、Python
    一行程序），不是几句定性描述糊弄过去。
  - Makefile 新增 `check-pollution`/`render-docs`/`demo` 三个目标。
  - 单元测试 35 条（tests/unit/a12/：gateway app 9、incident_log 11、
    render_docs 10、audit_pollution 3、demo 4）、元测试 4 条
    （tests/meta/a12/，证明 pollution audit 能真的报红且不是永远红）、
    烟测 2 条（tests/smoke/a12/：门禁 REJECT → incident-log 计数
    全链路、render_docs+audit_pollution 对 run 发现结果的一致性——
    子进程版的完整 `make demo` 流程改为人工跑通两次验证，不在 pytest
    里重复跑，避免子进程/端口相关的偶发脆弱性）。`make verify-a12`
    （走泛用 `verify-%`模式规则）+ 三个新增 Makefile 目标
    （check-pollution/render-docs/demo）全部跑通；mypy --strict 对
    92 个源文件干净；全项目 604 个测试全绿，无跨轮 regressions；
    `check_no_cheating`/`check_placeholders` 均干净。

遇到什么问题：
  1. 写 A12 之前重新读了一遍已交付的 A9/A10 代码，发现"拦截记录
    append-only 写 incident-log"这句话在两轮提示词里都出现过，但
    `gate/admission.py`和`release/rollback.py`里都没有真正调用任何
    文件写入——是两轮实现时都遗漏的真实缺口，不是这轮凭空加的新
    需求。选择在 A12 补上而不是往回改 A9/A10 的 commit：A9/A10 的
    核心决策逻辑本身没错（门禁判 REJECT、回滚触发都对），缺的只是
    "写日志"这一步，用一个新模块 + 调用方显式调用来补，比回头改两个
    已经交付、已经全测过的 round 侵入性更小。
  2. 类似地发现 A6 的六个网关模块和 A7 的 `GatewayMetrics` 之间、以及
    A6 六个模块彼此之间，都从未被证明能真正组合工作——每个模块的
    单元测试都是孤立断言，没有一处端到端验证过"一个真实请求经过
    全部六道关卡"。补 `gateway/app.py` 时顺带发现配额检查的真实语义
    只能是生成后（不是 PLAN.md 字面暗示的生成前）——见 DD-0027，
    没有为了凑字面意思去改两轮之前已交付的 `quota.py` 接口。
  3. `gateway/app.py`的`completions`处理函数最初写成一个大函数，ruff
    的 C901 圈复杂度检查报了 16（CLAUDE.md §9 要求 ≤10）——拆成
    `_authenticate_and_rate_limit`/`_resolve_backend`/`_generate`/
    `_enforce_quota` 四个独立函数后过检；拆分过程中发现一个真实
    bug：外层 `except Exception` 会连 `_resolve_backend` 故意抛出的
    `HTTPException`（路由到未配置模型时的 500）一起吞掉、错误地
    重新包装成"upstream model call failed"——加了一条显式
    `except HTTPException: raise` 直通分支修掉。
  4. `make demo`第一次手动全流程跑完后检查是否有遗留进程
    （`pgrep -f demo_stub_model_server.py`），确认子进程干净退出，
    没有僵尸进程或端口占用残留，才认为这一步真正做完，不是"脚本
    跑完没报错"就算数。
  5. 提前建好的 `tests/smoke/a12/` 目录一直是空的（建目录时只是为了
    跟其他轮次保持一致的三段式布局），第一次跑 `make verify-a12` 时
    pytest 因为"没收集到任何用例"直接以退出码 5 失败——补上两条真实
    的烟测（门禁 REJECT→incident-log 计数全链路、render_docs+
    audit_pollution 对 run 发现结果的一致性）而不是删掉这个空目录
    敷衍过去。

怎么解决的：见上。

测出什么数字：`docs/incident-log.md`里两条真实的门禁拦截记录（来自
两次真实运行 `make demo`）；`docs/crossover-curve.md`/
`docs/capacity-plan.md`里的并发数字（真实公式计算，估算输入，见
DD-0028）。除此之外 A12 本身不产生新的性能/准确率数字——它是收尾轮，
产出是文档、脚本、和补齐的两个真实缺口（incident-log 写入器、
gateway 应用层组装），不是新的 benchmark 结果。
```

```
B1（同一 session）—— A-track（推理/平台）全部 13 轮结束后，转入
B-track（训练）第一轮
做了什么：
  - train/gdn_lora_coverage.py —— 本轮头等大事：Qwen3.5-9B GDN 混合
    架构 LoRA 覆盖率硬验收。`parse_named_modules` 把真实
    `model.named_modules()`（或本沙箱的合成模块名）按层号分组，
    `LayerType`（FULL_ATTENTION/GDN/UNKNOWN）按该层实际出现过的子
    模块名后缀集合分类（不是硬编码层号，见 DD-0029）；
    `compute_trainable_coverage`/`assert_full_layer_coverage` 对着
    `target_modules` 算出逐层覆盖表并硬拒绝不满 32 层的配置；
    `CoverageReport.render_table()` 就是 PLAN.md 要求的"训练前打印
    逐层可训练参数分布"。
  - train/checkpoint_state.py —— 断点续训文件完整性。
    `REQUIRED_CHECKPOINT_FILES` 六个文件（真实 HF/PEFT 命名约定：
    `adapter_model.safetensors`/`optimizer.pt`/`scheduler.pt`/
    `rng_state.pth`/HF 自己的 `trainer_state.json`/本项目自己的
    `modelhub_trainer_state.json`，两个 trainer-state 文件为什么分开
    见 DD-0030）；`validate_checkpoint_complete` 缺文件即报错并点名
    哪个；`finalize_checkpoint` 先校验完整性再调用 A0 新增的
    `atomic_replace_dir` 做目录级原子落盘；`list_checkpoints` 只把
    完整的 checkpoint 计入（模拟崩溃留下的半截目录会被正确排除）。
  - common/atomic_io.py —— 新增 `atomic_replace_dir`（先把旧目录挪走、
    新目录 `os.replace` 到位、成功后再删旧目录；`os.replace` 本身不能
    覆盖非空目录，这是单文件原子写模式扩展到目录级别的必要变体）。
  - train/loss_guard.py —— `check_loss_finite` 对每个 step 的 loss 做
    NaN/Inf 检查，非有限值立即原子落盘完整 batch 上下文到
    `nonfinite_loss_step_<N>.json` 并抛出 `NonFiniteLossError`——不是
    跳过这个 step 继续跑，PLAN.md 原话"loss NaN/Inf 立即停止 dump"。
  - train/time_budget.py —— `TimeBudgetTracker`，`now_fn` 可注入
    （同 A6 `circuit_breaker.py` 先例），按 PLAN.md"按时间预算跑不按
    epoch 跑"驱动 `--max-hours`；`record_metric` 顺带追踪 best
    checkpoint，供"到点取最佳 checkpoint"用。
  - train/dry_run.py —— dry-run 三件套：`validate_dataset_schema`
    （第一条坏数据即硬失败，不是跳过统计）、
    `compute_token_length_stats`（含 p99，nearest-rank 算法）、
    `estimate_training_memory`/`assert_fits_in_memory`（显存预估，
    输入常数标注为示例值而非实测，见 DD-0031）。
  - train/sft_config.py —— `SftConfig`（`lr`/`batch_size`/`seed`/
    `max_seq_len`/`max_hours` 全部无默认值，CLAUDE.md §4）、
    `thinking_mode: bool` 必填且校验器强制为 False（PLAN.md 项 10）；
    `default_qwen35_gdn_target_modules` 就是 FFN∪attention∪GDN 后缀
    的并集；`render_llamafactory_yaml` 渲染真实 LLaMA-Factory 训练
    YAML 字段（`lora_target`/`finetuning_type`/`cutoff_len` 等），
    不是自造 schema——PLAN.md"不要重写训练循环"落到这一层就是"配置
    生成到位，CLI 调用交给 LLaMA-Factory 自己"。
  - train/mlflow_backend.py —— guarded `import mlflow`，本沙箱未装
    （`train` extra），`check_mlflow_available` 走 `CheckStatus`
    白名单模式；`start_mlflow_run` 未装时真实 `ImportError` 直接外抛，
    不假装启动成功。
  - train/callbacks.py —— `ModelHubTrainerCallback`，guarded 继承真实
    `transformers.TrainerCallback`（未装时退化成继承 `object`，纯粹
    为了本沙箱能 import 和单测，真实 GPU 机器上是真正的 HF 回调
    子类）；`on_log` 接 loss_guard、`on_save` 接
    checkpoint_state（校验+原子落盘）、`on_step_end` 接
    time_budget（预算耗尽即 `control.should_training_stop = True`）——
    PLAN.md 明确不让重写训练循环，三个实时检查全部通过 HF Trainer
    真实回调 API 挂载，不是自己再写一个 training loop。
  - train/runner.py —— `run_sft_preflight` 顺序跑五道检查（数据
    schema → token 长度分布 → GDN 覆盖率硬断言 → GDN 快 kernel 状态
    → 显存预估），任意一步失败立即停止（不像 A9 五道闸会跑完全部
    再汇总，preflight 是"能不能开始训"的门，不是训完之后的诊断报告，
    没有必要在已经确定不能开始的情况下继续跑后面的检查）；
    `check_llamafactory_available`/`run_sft_training` 真实调用
    `llamafactory-cli train` 子进程，`timeout=None` 是显式且有注释
    说明的选择（真实时长由 `max_hours` 通过回调控制，不是外部
    subprocess 超时该管的事）。
  - configs/train/ 三个真实 YAML：`gdn_lora_coverage.yaml`（32 层 +
    11 个 target_modules）、`memory_estimate.yaml`（A100-80G 预算，
    常数来源见 DD-0031）、`sft_qwen3_5_9b.yaml`（完整 `SftConfig`，
    `thinking_mode: false`）——三份都验证过能被 `load_yaml_config`
    真实加载并算出合理数字。
  - 单元测试 94 条（tests/unit/b1/：gdn_lora_coverage 12、
    checkpoint_state 11、dry_run 13、loss_guard 5、time_budget 9、
    smoke_test 12、sft_config 9、mlflow_backend 3、callbacks 8、
    sft_runner 12）、元测试 6 条（tests/meta/b1/：GDN 覆盖率硬门禁
    的朴素配置真实触发拒绝 + 修复后真实通过、smoke-test 三项验收
    标准逐一真实触发失败 + 全通过场景不是永远红）、烟测 1 条
    （tests/smoke/b1/：20 条样本规模跑完整 preflight 五道检查链，
    CLAUDE.md §1.4 只缩规模不换实现）。`make verify-b1`（走泛用
    `verify-%` 模式规则，无需新增 Makefile 目标）三段全绿；
    mypy --strict 对 102 个源文件干净；全项目 711 个测试全绿，
    无跨轮 regressions；`check_no_cheating src`/`check_placeholders
    docs src` 均干净（2 条历史遗留 allowlist 项与本轮无关）。

遇到什么问题：
  1. 设计阶段自查发现 `checkpoint_state.py` 最初把本项目自己的续训
    状态文件命名为 `trainer_state.json`——和 HuggingFace `Trainer`
    自己用来续训的文件完全同名，写入会静默覆盖 HF 自己的续训关键
    文件。这个 bug 只有真实 GPU 上跑 `--resume-from` 才会暴露，写
    测试之前先发现并改成 `modelhub_trainer_state.json`，同时把 HF
    自己的文件列成 `HF_TRAINER_STATE_FILENAME` 一起纳入必需文件集合
    （DD-0030）。
  2. 同一次自查还发现 RNG 状态文件名最初写成 `rng_state.pt`，核对
    HF Trainer 真实产物后确认约定是 `.pth`，在写测试前改正——否则
    `validate_checkpoint_complete` 会对着一个真实 HF checkpoint 目录
    报"缺文件"，因为文件名本身就没对上。
  3. `runner.py` 里 `subprocess.run(..., timeout=None, ...)` 一开始
    担心会被 `check_no_cheating.py` 的 NO_TIMEOUT_CALL 检查当成"故意
    传 None 绕过超时检查"——核实检查器只要求出现 `*timeout*` 关键字
    参数、不检查值是否非 None，语法上能过；但为了让这个选择读起来
    是"深思熟虑"而不是"钻检查器空子"，加了一段注释说明为什么 None
    在这里是唯一正确的值（真实时长由 `max_hours` 通过训练循环内部
    的回调控制，外部 subprocess 级别的固定超时会错误地杀掉一个正在
    合法运行的多小时训练任务）。
  4. `smoke_test.py` 的报错信息最初直接引用 PLAN.md 原文里带全角逗号
    "，"的中文句子，ruff 的 RUF001（歧义全角标点）在字符串字面量里
    命中（项目 `ignore = ["RUF002","RUF003"]` 只覆盖另外两条，不包括
    RUF001），改成纯英文措辞而不是加 `noqa` 绕过。
  5. `callbacks.py` 的 guarded 继承模式（`transformers.TrainerCallback`
    未装时退化成 `object`）在把 `mlflow.*` 加进 mypy overrides 之后，
    之前两处 `# type: ignore[assignment,misc]` 变成"未使用的
    ignore"报错；删掉后 mypy strict 的 `disallow_subclassing_any`
    又在类定义那一行真实报错（"继承一个类型是 Any 的基类"）——加回
    一条精确到那一行的 `# type: ignore[misc]`，而不是在整个文件顶部
    加一条宽松忽略。
  6. `tests/unit/b1/test_runner.py` 和已有的 `tests/unit/a4/
    test_runner.py` 撞了 basename——这是本 session 第三次撞到同一类
    问题（A11 时也撞过一次），改名成 `test_sft_runner.py`，同时把这
    条检查确认写进了自己的例行 pre-commit 习惯
    （`find tests/unit tests/meta tests/smoke -name "test_*.py" |
    xargs -n1 basename | sort | uniq -d`）。
  7. `run_sft_preflight` 里的 GDN 快 kernel 状态检查
    （`detect_gdn_kernel_status`）在本沙箱（无 CUDA 设备）永远是
    `degraded=True`——这本身是真实、诚实的检测结果，不是 bug；但这
    意味着"健康路径"的单元/元/烟测必须显式 monkeypatch 这一步的
    结果，否则每一个原本用来测别的东西（GDN 覆盖率、显存预估、
    schema 校验）的用例都会先被这一道无关的真实红灯挡住。专门留了
    一条不 monkeypatch 的用例
    （`test_degraded_kernel_status_refuses_to_start`）验证这道真实
    检测本身的诚实性没有被后续开发不小心破坏掉。

怎么解决的：见上。

测出什么数字：无（无 GPU，本轮不产生任何训练/显存/loss 的真实测量
数字）。`estimate_training_memory` 在 batch_size=4/max_seq_len=4096
下算出的 headroom≈59.6% 是公式正确性的验证，输入常数是示例值不是
实测值（DD-0031）——这条数字本身不得被任何简历或报告引用，只能引用
"公式跑通、待真机替换输入"这个工程结论。

诚实清单：
  - 没做：真实模型加载/真实 32 层 `named_modules()` 探测（本沙箱无
    GPU、无法下载 Qwen3.5-9B 权重）——`gdn_lora_coverage.py` 的所有
    测试都基于按本项目文档理解合成的模块名树，不是真实探测结果；
    真实 50 步烟测（`smoke_test.py` 只有比对/判定逻辑，没有真实
    显存峰值/真实续训曲线可喂）；真实 LLaMA-Factory 训练子进程
    调用（`llamafactory-cli` 未装，`run_sft_training` 在本沙箱里
    唯一被测试到的路径是"未安装时正确拒绝"）；真实 MLflow 落盘。
  - 假设了什么：Qwen3.5-9B 的 32 层里 GDN 层的子模块后缀命名是
    `in_proj_qkvz`/`in_proj_ba`/`out_proj`/`conv1d`（这是本项目对
    Qwen3-Next 系列架构文档的最佳理解，未经真实模型验证，配置文件
    与代码注释里都显式标注这一点）；全注意力层与 GDN 层在真实模型
    里互斥、不存在混合层（DD-0029 的"什么情况会失效"一节）；
    `estimated_activation_bytes_per_token`/`lora_trainable_param_bytes`
    两个显存预估常数（DD-0031）。
  - 已知局限：`ModelHubTrainerCallback` 的三个真实回调方法
    （`on_log`/`on_save`/`on_step_end`）从未在真实 HF `Trainer` 实例
    上跑过一次完整训练循环——单测里全部用手写的 fake `args`/
    `state`/`control` 对象验证签名和行为，真实 HF Trainer 传入的
    对象结构是否完全吻合，只有真机训练才能最终确认。
```

```
B2（同一 session）
做了什么：
  - train/experiments/ 新包（六组对比编排，PLAN.md v2 微调版）：
    `common.py`（`TrainingRunMetrics`/`LayerTypeProfileSplit`——v2 新增
    的 GDN 层 vs 全注意力层显存/耗时占比字段——`compute_throughput_
    tokens_per_s`/`compute_total_cost` 纯公式、`check_training_
    experiment_precondition` 复用 A11 `assert_not_polluted` 同款一行、
    `guard_dedicated_gpus` 复用 A8 `bench/gpu_guard.py::
    guard_gpu_exclusivity` 按第4组"双卡"逐张检查）。
  - `metrics_recorder.py` —— 本项目自己控制的 `experiment_metrics.json`
    schema（不去反推 HF `trainer_state.json` 里没保证记录的字段），
    `ExperimentMetricsCallback` 复用 B1 `callbacks.py` 的 guarded-继承
    模式记录真实 wall-clock 单步耗时 + `torch.cuda.max_memory_
    allocated()`。
  - `peft_method_comparison.py`（组1-3：LoRA r=32/QLoRA 4bit bnb/
    全参+ZeRO-3）—— 真实 LLaMA-Factory YAML 字段区分三法
    （`finetuning_type`/`quantization_bit`+`quantization_method`/
    `deepspeed`），`trainable_param_count_for_method` 纯函数复现
    PLAN.md 说的"LoRA/QLoRA 只训 adapter，全参训全部"。
  - `distributed_strategy_comparison.py`（组4：ZeRO-2/ZeRO-3/FSDP
    双卡）—— 真实 DeepSpeed ZeRO JSON 配置 + 真实 Accelerate FSDP
    配置 schema 都写了；FSDP 通过 `llamafactory-cli` 具体怎么拉起
    这一步的调用形态本项目没有足够把握确认，`run_distributed_
    strategy_comparison_group(FSDP)` 显式 `NotImplementedError`（同
    A11 GPTQ 校准集缺口的处理方式），不是编一个看起来合理的命令行。
  - `kernel_optimization_comparison.py`（组5-6：Liger Kernel on/off、
    Flash Attention on/off、序列打包 on/off——读作三个独立开关而非
    2x2 交叉，对应 PLAN.md"各200步"的措辞）——`KernelTogglePair`
    算三项 delta 百分比。
  - `report.py` —— PLAN.md 两条硬性框架规则直接写成代码而不是留给
    写报告的人记住：`format_effect_cost_sentence` 从不出现"消融显示"
    字样；`accuracy_delta_points` 是必填参数（不是可选默认 None 悄悄
    省略），传 `None`（本沙箱唯一诚实状态）会生成一句显式的"尚无收敛
    跑数据"，而不是一句从不提准确率的句子。句子里的中文一律用 ASCII
    标点（不用全角，，。：）——ruff 的 RUF001 对全角标点在真实字符串
    字面量里报错，项目里 B1 `smoke_test.py`已经定下"改写而不是加
    suppression"的先例，这里沿用。
  - `orchestrator.py` —— 直接复用 A11 `bench/experiments/common.py`
    的 `ExperimentGroupOutcome`/`run_experiment_group_isolated`（本来
    就是对"一组是什么"泛化的，训练侧原样能用），没有为训练侧重新
    发明一套等价的隔离边界。
  - `configs/train/experiments/` 三份 YAML —— `gpu_cost_per_hour`
    刻意不作为字段出现（DD-0032），其余字段与 `configs/train/
    sft_qwen3_5_9b.yaml` 保持一致（PLAN.md"严格控制变量"）。全部
    验证过能被 `load_yaml_config` 真实加载。
  - 单元测试 81 条（tests/unit/b2/：experiments_common 25、
    metrics_recorder 7、peft_method_comparison 14、distributed_
    strategy_comparison 15、kernel_optimization_comparison 10、
    comparison_report 8、comparison_orchestrator 4——两个文件改名
    避免与 a4/a11 撞 basename）、元测试 5 条（tests/meta/b2/：共享
    precondition 通过真实 orchestrator 入口真实触发拒绝 + 不是永远
    红；一个真实的 llamafactory-cli 不可用失败真实隔离、不拖累同批
    次的兄弟组）、烟测 1 条（tests/smoke/b2/：两组对比走完整
    orchestrator→report 链路，缩小到 2/6 组）。`make verify-b2`
    （走泛用 `verify-%` 模式规则，无需新增 Makefile 目标）三段全绿；
    mypy --strict 对 110 个源文件干净；全项目 800 个测试全绿，无
    跨轮 regressions；`check_no_cheating src`/`check_placeholders
    docs src` 均干净。

遇到什么问题：
  1. 第一版 `configs/train/experiments/peft_method_comparison.yaml`
    给 `gpu_cost_per_hour` 写了个"看起来合理"的 `15.0`——写完之后
    核对 FACTS.md 才发现 2×A100-80G 租卡成本全项目唯一有的数字是一个
    "行情波动，以实际报价为准"的未确认区间，而 A8 的
    `bench/cost_model.py` 早就为同类情况（RTX 5090 无确认价）定过
    先例：把这类参数做成必填调用参数、从不做模块默认值。三处
    `ExperimentConfig` 类和三份 YAML 都改成不含这个字段，
    `run_*_comparison_group` 加一个必填 `gpu_cost_per_hour` 关键字
    参数（DD-0032）。
  2. `metrics_recorder.py::_real_peak_memory_bytes` 第一版在
    `except ImportError` 分支写了 `return 0`——写完全部代码后按例行
    习惯跑 `scripts/check_no_cheating.py src`，真的报了一条
    `[EXCEPT_RETURN_CONSTANT]`：这正是 CLAUDE.md §1.1/§2 点名禁止的
    "缺失指标补 0"模式本身，不是误报。改成 `int | None`、缺测返回
    `None`，新增 `require_measured_peak_memory` 作为下游构造
    `TrainingRunMetrics`（该类型自己的 `peak_memory_bytes` 仍是必填
    `int`——这是四个核心数字之一，不允许缺）前的硬失败关卡
    （DD-0033）。这是本轮最有价值的一次自查——检查器不是摆设，这次
    真的抓住了一处会污染下游对比数字的真实 bug。
  3. FSDP 通过 LLaMA-Factory 具体怎么拉起（是走 `accelerate launch`
    包一层，还是 `FORCE_TORCHRUN` 之类的环境变量，还是别的机制）
    本项目没能查到足够确定的依据——没有编一个看起来合理但可能是错的
    命令行，`run_distributed_strategy_comparison_group(FSDP)` 显式
    `NotImplementedError`，Accelerate FSDP 配置 schema 本身仍然
    照常写（这部分有把握），只把"怎么调用"这一步留白。
  4. B2 第三次撞上 test basename 冲突（`test_common.py` 撞 A11、
    `test_report.py` 撞 A4）——继续沿用例行 pre-commit 检查命令，
    改名为 `test_experiments_common.py`/`test_comparison_report.py`，
    顺带把 `test_orchestrator.py`（会撞 A10）提前改名为
    `test_comparison_orchestrator.py`，没有等冲突真的发生再改。

怎么解决的：见上。

测出什么数字：无（无 GPU，六组对比一条真实短跑都没能执行——本沙箱
连 `llamafactory-cli` 都没装）。烟测里的显存/吞吐/成本数字全部来自
手写的合成 `experiment_metrics.json`，只验证 orchestrator→report 链路
本身接得通，不是任何真实短跑的结果，不得被任何报告或简历引用。

诚实清单：
  - 没做：任何一组的真实 200/300 步短跑（LLaMA-Factory 未装、无 GPU）；
    `communication_time_ratio` 的真实测量（这个模块从不自己测通信
    时间，只在 `RawExperimentMetrics` 里留了一个可选字段，等外部真实
    NCCL/DeepSpeed trace 数据填入）；`LayerTypeProfileSplit`（v2 新增
    的 GDN vs 全注意力层显存/耗时占比）从未在任何 `run_*_comparison_
    group` 里被真实填充过——三个函数返回的 `TrainingRunMetrics.
    layer_type_split` 全部是 `None`，这个字段目前只有它自己的单元
    测试（构造+校验+占比计算）验证过逻辑正确，没有一条真实的采集
    路径把它接到 `ExperimentMetricsCallback` 里。
  - 假设了什么：LLaMA-Factory 的 `flash_attn`/`packing`/`enable_liger_
    kernel`/`quantization_bit`+`quantization_method`/`deepspeed` 几个
    YAML 字段名是这个项目当前查到的、认为正确的写法，未在真实
    LLaMA-Factory 安装上验证过（同 B1 `sft_config.py` 的诚实标注）；
    ZeRO-2/ZeRO-3 用 `FORCE_TORCHRUN=1`+`NPROC_PER_NODE` 环境变量拉起
    多卡是本项目对 LLaMA-Factory 文档的最佳理解，同样未验证。
  - 已知局限：六个对比组从未被证明能在真实环境里端到端跑通哪怕一次
    ——每个 `run_*_comparison_group` 在本沙箱里唯一被测到的分支是
    "llamafactory-cli 不在 PATH 上，正确拒绝"，真实的"subprocess 跑完
    →真实 experiment_metrics.json 写出→真实 TrainingRunMetrics 构造"
    这条主链路只在烟测里用合成数据模拟过，没有一次是对着真实子进程
    输出走过的。
```

```
B3（同一 session）—— 本轮与 A9/A12 现成机制高度重合，规模明显小于
B1/B2，按实际范围收敛（3个库模块 + 1个真实脚本，不是又一整套编排包）
做了什么：
  - train/bad_models/ 新包：`profiles.py`（三个 profile 的静态元数据——
    怎么造/期望被哪道闸拦/为什么，ckpt-D 的 expected_gate 诚实写成
    "accuracy（不是 PLAN.md 字面的 safety）"，见 DD-0034）、
    `dataset_filters.py`（`filter_easy_only`——Spider easy + BIRD
    simple，两个源各自的难度词汇不做跨数据集映射；`inject_crud_
    samples`——复用 A9 safety_gate 同款 `Source.MINIDEV_CRUD` 校验）、
    `training_configs.py`（ckpt-B/ckpt-D 的 200 步 LoRA YAML 渲染，
    ckpt-A 无训练配置——PLAN.md"零成本"直接选早期 checkpoint）、
    `checkpoint_selection.py`（`select_underfit_checkpoint`——真实
    B1 `list_checkpoints` 输出里选最早一个）、`adapter_export.py`
    （"只导出 LoRA adapter，不传合并权重"的真实可测门禁——检测到
    `model.safetensors`/`pytorch_model.bin` 等合并权重文件存在时
    硬拒绝导出，不是导出时静默忽略）。
  - scripts/bad_model_drill.py —— 真实脚本（同 A12 `demo.py` 的规格：
    真实数据 + 真实 A9 五道闸 + 真实 incident-log 写入，唯一不真实的
    是"checkpoint 本身"，本沙箱没有 GPU 训不出来）。手动完整跑通一次：
    ckpt-A（4/20 对，真实 20% 准确率）→REJECT(accuracy)；ckpt-B
    （baseline 全对，candidate 10 道复杂题全错）→REJECT(regression)，
    `find_regressions` 真实点名 10 个 sample_id；ckpt-D 两段式——(a)
    真实 `check_safety_gate` 对两条真实 CRUD 对抗样本跑出 PASS（沙箱
    强制执行本身没坏），(b) 真实 `execute_isolated('DELETE...')` 被真实
    拦下（`UNSAFE_STATEMENT`），6 条这样的预测拉低准确率到 30%→
    REJECT(accuracy)。三条真实 GATE_REJECTION 记录写进
    `docs/incident-log.md`，两份真实训练 YAML 写进
    `artifacts/bad_models/configs/`，`artifacts/bad_models/README.md`
    从三个 profile 的真实 drill 结果生成，不是手打的静态文案。
  - Makefile 新增 `bad-model-drill` 目标（同 `demo` 目标的调用形状）。
  - 单元测试 32 条（tests/unit/b3/：profiles 5、dataset_filters 8、
    training_configs 4、checkpoint_selection 2、adapter_export 5、
    bad_model_drill 9——脚本的每个辅助函数都对着真实 tmp SQLite db
    测过，不只测"构造函数不报错"）、元测试 4 条（tests/meta/b3/：
    三个 profile 的真实失败信号各自真实触发 REJECT + 一个健康候选
    走同一条真实 gate 路径证明不是永远红）、烟测 1 条（tests/smoke/b3/：
    三个 profile 全部跑完 + incident 构造 + README 渲染端到端一条链，
    不重复调用真实 `main()`——同 A12 demo.py 的先例，避免每次跑测试
    都往真实 `docs/incident-log.md` 追加记录）。`make verify-b3`
    （走泛用 `verify-%` 模式规则）三段全绿；mypy --strict 对 116 个
    源文件干净；全项目 837 个测试全绿，无跨轮 regressions；
    `check_no_cheating src`/`check_placeholders docs src` 均干净。

遇到什么问题：
  1. PLAN.md ckpt-D 原文说"期望被 GATE_SAFETY 拦"，写 profiles.py 之前
    先读了一遍 `gate/safety_gate.py` 的真实实现和它自己的文档字符串，
    发现这道闸门是平台自身只读强制执行的检查，"甚至不看候选模型会
    生成什么"，和 PLAN.md 字面暗示的"候选模型的不安全 SQL 被这道闸
    拦下"完全是两回事。没有为了凑字面描述去改 `check_safety_gate` 的
    语义（那会破坏 A9 自己"平台完整性"这个独立检查维度），而是真的
    跑了一遍脚本验证——喂真实 CRUD 对抗样本给真实安全闸，得到的确实
    是 PASS 不是 REJECT；ckpt-D 真正会被拦下的机制是它自己预测的
    DELETE 语句被真实 sqlexec 拦截（`UNSAFE_STATEMENT`），这类预测算
    错误答案拉低 `execution_accuracy`，实际触发的是 accuracy 闸
    （DD-0034）。这是本轮最有价值的发现，一半的价值就在"读代码验证
    PLAN.md 的字面描述是否成立"这件事本身。
  2. 一开始考虑给 B3 单独建一套类似 A11/B2 的 `experiments/`/编排包
    结构，写完 profiles.py 之后重新评估发现 B3 实际复用面很大——三个
    admission gate、CRUD 对抗集约定、checkpoint 文件约定、incident
    log 写入器全部是 A9/A1/B1/A12 现成的，B3 真正新增的只是"怎么造出
    这三种坏信号 + 怎么导出干净的 adapter"，按实际新增内容收窄成
    3 个小模块 + 1 个真实脚本，没有为了显得"完整"硬凑一个不必要的
    子包层级。
  3. `export_adapter_only` 的"不传合并权重"检测最初只想在文档字符串
    里提一句，写测试时意识到这必须是真实、可测的运行时检查——
    `test_merged_weight_file_present_is_rejected` 真的往一个完整
    checkpoint 目录里塞一个 `model.safetensors`，验证导出函数会真的
    拒绝而不是静默忽略多出来的文件。

怎么解决的：见上。

测出什么数字：`docs/incident-log.md` 里三条真实的门禁拦截记录（来自
一次真实跑通的 `make bad-model-drill`）——ckpt-A 20.00% 准确率、
ckpt-B 10 个真实点名的回归 sample_id、ckpt-D 30.00% 准确率（含
UNSAFE_STATEMENT 拉低效应）。这些数字全部来自本轮设计的、明确标注
"engineered-but-real" 的演练用预测集，不是真实训练出的 ckpt-A/B/D
的准确率——不得被当作任何真实模型的性能数字引用，只能引用"门禁演练
机制本身跑通、产出了真实拦截记录"这个工程结论。

诚实清单：
  - 没做：ckpt-A/B/D 三个 adapter 的真实训练（本沙箱无 GPU，
    `training_configs.py` 只写出了 ckpt-B/ckpt-D 的真实 LLaMA-Factory
    YAML，ckpt-A 按 PLAN.md 要求走"直接选早期 checkpoint"零训练路径，
    但 `checkpoint_selection.select_underfit_checkpoint` 从未对着一个
    真实 B1 checkpoint 目录跑过——本沙箱没有真实 B1 训练产出）；
    `export_adapter_only` 从未对着一个真实 LoRA adapter 导出过（单测
    用的是 B1 checkpoint_state.py 同款合成 fixture 文件）。
  - 假设了什么：`adapter_config.json` 是真实 PEFT/LLaMA-Factory 会
    产出的文件名（`ADAPTER_ONLY_EXPORT_FILES` 里把它列为"存在就导出，
    不存在不强制"，因为 B1 的 checkpoint_state.py 从未真正建模过这个
    文件，这是本轮基于 PEFT 通用约定的补充假设，未经真实训练验证）。
  - 已知局限：`scripts/bad_model_drill.py` 里 ckpt-A/B/D 的"预测集"
    是本轮为了复现 PLAN.md 描述的失败信号手工设计的，不是真实模型
    推理出来的——真实 ckpt-A/B/D 在真机上训出来后，它们的真实预测
    分布是否恰好落在这里设计的信号形状里（比如真实欠拟合模型是不是
    真的低于 50% 准确率、真实回归是不是恰好只出现在"复杂"难度切片），
    只有真机训练+真机评估才能最终确认。
```

```
B4（同一 session）—— B-track 训练线继续，本轮不新建包，单文件
`src/modelhub/train/dpo.py`，按 PLAN.md 原文字面要求
做了什么：
  - train/dpo.py —— 本轮真正的重量在偏好对构造（PLAN.md 步骤1-7），
    不在 DPOTrainer 接线：
    - `classify_prediction` 白名单式穷举分类：`EXEC_OK_CORRECT`（跑通
      且对）/`EXEC_OK_INCORRECT`（跑通但错）/`EXEC_FAILED`（PLAN.md
      字面的"语法错"——SYNTAX/SEMANTIC/TIMEOUT，外加 PLAN.md 原文没提
      但同样明确是模型自己的错、明确比 EXEC_OK_INCORRECT 更差的
      UNSAFE_STATEMENT，一起归进这一档而不是留着不处理）；对任何
      未预期的 exec_code 硬抛异常，不做兜底分类（CLAUDE.md §1.3）。
    - ★★ 系统错（HARNESS_DB_UNAVAILABLE/HARNESS_INTERNAL）/
      UNDECIDABLE/OUTPUT_TRUNCATED 三类整条丢弃，`ClassifiedCandidate.
      tier`为`None`、`discard_reason`必须非空（构造函数互斥校验）——
      永远不会出现在任何一对偏好对的任意一侧。
    - `build_preference_pairs_for_question`：只在不同 tier 之间配对
      （同级内不配对），把每个问题内所有 kept 候选两两比较、tier 严格
      更优的一方当 chosen；`PreferenceDatasetReport` 聚合
      pair_count/tier_counts/discard_counts/exec_ok_correct_share。
    - `assess_expected_dpo_benefit`：PLAN.md ★"如果「跑通且对」占比
      已经很高，DPO 收益会很小——这个观察本身写进决策记录，比硬跑一遍
      有价值"——把这句话做成真实计算 `exec_ok_correct_share` 并按
      阈值给结论的函数，不是只在文档里定性描述（DD-0035）。
    - `sample_k_candidates`：PLAN.md 步骤1"对 2000 个问题各采样
      k=4 个 SQL"——直接复用 A4 `eval/runner.py::run_one_sample`
      跑 k 次，没有为 B4 重新写一遍生成/执行/比对逻辑。
    - `DpoSamplingConfig`（k/温度/max_tokens/timeout，PLAN.md 步骤7
      "采样温度和 k 记进 manifest"——这些字段本身就会被
      `compute_config_hash` 哈希进 manifest 的 `config_hash`，不需要
      manifest schema 专门加字段）、`DpoTrainingConfig`（`lora_
      target_modules`沿用 B1 全 32 层覆盖配置，PLAN.md ★ 步骤8）。
    - `check_trl_available`/`build_dpo_config_kwargs`（真实 TRL
      `DPOConfig`字段名，纯函数、不需要装 trl 就能测）/
      `run_dpo_training`（guarded：trl 未装则真实拒绝；trl 已装的
      分支目前是显式 `NotImplementedError`——`DPOTrainer`构造需要
      真实加载的 model/tokenizer/`datasets.Dataset`，本沙箱一样都
      没有，没有编一个未经验证的调用形状去凑"能跑"的假象）。
    - "训完跑快评"（PLAN.md 步骤末）：直接复用 A4 `run_eval` +
      快评固定子集，没有为 B4 写一个只是转发参数的包装函数。
  - configs/train/dpo_sampling.yaml / dpo_training.yaml —— 两份真实
    YAML，均验证过能被 `load_yaml_config` 正确加载；`beta=0.1`
    标注为"TRL 自己常引用的默认值，本项目未调参"，不是隐式继承库
    默认值。
  - pyproject.toml —— `train` extra 新增 `trl>=0.9`，mypy overrides
    新增 `trl.*`。
  - 单元测试 26 条（tests/unit/b4/：`classify_prediction` 对每一个
    真实 `exec_code`/`comparison_result` 组合穷举测过，包括"未预期
    exec_code 硬抛异常"这一条；`sample_k_candidates` 用真实 SQLite db
    + 真实 sqlexec/compare + 假 ModelClient 跑通 k=4 全流程）、元测试
    4 条（tests/meta/b4/：系统错/UNDECIDABLE/截断三类样本即使是"唯一
    替代候选"也真实不会进任何一对；同级四个候选真实产出零偏好对；
    健康的混合 tier 输入真实产出非零偏好对，证明机制不是永远零对）、
    烟测 1 条（tests/smoke/b4/：5 题×k=4，真实 sqlexec/compare 全程
    跑通到 `PreferenceDatasetReport`+benefit 评估，缩小题量不换实现）。
    `make verify-b4`（走泛用 `verify-%` 模式规则）三段全绿；
    mypy --strict 对 117 个源文件干净；全项目 868 个测试全绿，无
    跨轮 regressions；`check_no_cheating src`/`check_placeholders
    docs src` 均干净。

遇到什么问题：
  1. `classify_prediction`最初用一个共享 `common: dict` + 双星号展开
    构造 `ClassifiedCandidate`，mypy --strict 在多处报"参数类型不
    兼容"（`dict[str, object]`展开丢失了每个字段各自的精确类型）。
    改成一个局部闭包函数 `_classified(*, tier, discard_reason)`
    显式传参，类型检查干净，代码量没有明显增加。
  2. `PredictionRecord.exec_code`真实取值空间比 PLAN.md 三档描述
    （跑通且对/语法对但结果错/语法错）多一种——`execute_isolated`对
    predicted_sql 里出现 CRUD 语句时会返回 `UNSAFE_STATEMENT`（A9
    安全闸同一机制），PLAN.md 原文没提这种候选该归哪一档。判断
    "模型自己生成了危险语句"明确比"跑通但结果错"更差、又不是系统错
    不该丢弃，归进 PLAN.md 字面的"语法错"档（重命名成更准确的
    `EXEC_FAILED`，docstring 里显式说明这不是字面意义的语法错误）。
  3. 写 `assess_expected_dpo_benefit` 时确认了这条判断该落在哪：
    PLAN.md 原文说"这个观察本身写进决策记录，比硬跑一遍有价值"，
    没有直接说"写成代码"——但既然决策记录要有数字支撑（CLAUDE.md
    §12"不确定就说不确定"的反面是"确定的就该有真实数字"），把这个
    观察做成真实计算的函数、DD-0035 引用它手写的示例性验证数字
    （75% share，20 题×k=4 的合成配比），而不是在决策记录里凭空写
    "预计会很高"。

怎么解决的：见上。

测出什么数字：无（无 GPU，本轮不产生任何真实 DPO 偏好对或训练数字）。
DD-0035 里的 `exec_ok_correct_share=75.0%`/`pair_count=60` 来自本轮
为验证 `assess_expected_dpo_benefit` 可用而手写的示例性合成候选集
（20 题、k=4、3:1 正确:错误配比），不是任何真实 SFT 模型的采样结果，
不得被当作 B1 真实模型的准确率引用。

诚实清单：
  - 没做：任何真实的 2000 题×k=4 采样（本沙箱没有真实 served 模型，
    `HttpModelClient`未在这里被真实调用过）；真实 `DPOTrainer`
    构造与训练（`run_dpo_training`目前对已安装 trl 的分支是显式
    `NotImplementedError`，真实调用形状留给真机验证）；"训完跑快评"
    这一步从未真实执行过（没有真实 DPO 训练产物可评）。
  - 假设了什么：`UNSAFE_STATEMENT`归入 `EXEC_FAILED`档（PLAN.md
    原文未提及这种候选，本轮基于"比跑通但错更差、不是系统错"的判断
    自行归类，见上）；0.7 的 `exec_ok_correct_share`阈值是本项目
    自定、未经真实数据验证的判断线（DD-0035"什么情况会失效"一节）。
  - 已知局限：三级配对策略（同一问题内所有 kept 候选两两跨 tier
    全配对，不是只取 top-1 vs rest）是本轮的设计选择，PLAN.md 没有
    明确这一层实现细节该是全配对还是仅相邻 tier 配对——真机第一批
    真实偏好对里，如果某些问题的候选高度集中在两三个 tier、全配对
    产生的偏好对数量是否会因为重复样本对权重失衡影响训练效果，需要
    真机训练+效果验证才能确认，本轮只保证"不同级不配对"这条硬约束
    被遵守。
```

```
B5（同一 session）—— B-track 训练线收尾，PLAN.md v2 微调版
做了什么：
  - train/grpo/ 新包（PLAN.md 明确要求目录形态"src/modelhub/train/
    grpo/"，不是单文件）：
    - `reward.py` —— 本轮真正的难点（PLAN.md 原话"本轮真正的难点，
      也是最好的面试素材"）：`compute_reward` 白名单式穷举分类，
      mask（`reward=None`，永远不是 0.0）覆盖 PLAN.md ★ 点名的
      HARNESS_DB_UNAVAILABLE/HARNESS_INTERNAL/OUTPUT_TRUNCATED，
      外加本轮自己判断该同样 mask 的 UNDECIDABLE（DD-0036，与 B4
      `dpo.py` 对同一类样本的处理保持一致）；`RolloutReward` 用
      `__post_init__`强制"MASKED 必须 reward=None+有 mask_reason，
      非 MASKED 必须有真实 reward+无 mask_reason"这条互斥约束，不是
      靠调用方自觉维护。
    - `group_diagnostics.py` —— PLAN.md ★"reward 全0或组内全相同→
      告警"：`diagnose_group`判定"全相同"覆盖全 0/全 1/任意其他
      单一值（不是只判 0 这个特殊情况），外加"全部被 mask、组内没有
      任何可比较的 reward"这个 PLAN.md 没直接点名但同样致命的退化
      场景（advantage 同样算不出来）。
    - `step_diagnostics.py` —— PLAN.md ★ 项3-4"每步记录系统错/超时/
      截断占比+reward直方图"+"系统错占比>阈值→中断训练并告警"：
      中止阈值直接复用 `eval/metrics.py::HARNESS_ERROR_FLOOD_
      THRESHOLD`（1%，CLAUDE.md §2.2 原文"占比>1%→中止"），不是给
      GRPO 单独发明第二个数字。
    - `rollout_timing.py` —— PLAN.md 项6"★ 测 rollout 时间里有多少
      花在等 SQL 上"：并发+超时机制直接复用 A2 `sqlexec/pool.py::
      execute_many`（不重新实现），这个模块只加了真实 wall-clock
      计时包装，让"等 SQL 占比"是真实测量值不是估算。
    - `kernel_guard.py` —— PLAN.md ★ 项9"rollout 侧 vLLM 同样要验证
      GDN kernel 状态"：直接复用 B1/A5 `serve/kernel_status.py::
      detect_gdn_kernel_status`，没有为 rollout 侧重新写一遍检测
      逻辑。
    - `smoke_test.py` —— PLAN.md ★"先跑10步烟测确认显存与流水线"：
      显存检查复用 B1 `train/smoke_test.py::check_peak_memory`原样，
      流水线完成度是 GRPO 自己的新标准（目标步数 vs 实际完成步数 +
      是否被系统错占比中止）。
    - `runner.py` —— `GrpoTrainingConfig`、`check_verl_available`
      （guarded，本沙箱未装 verl）、`run_grpo_preflight`（目前只接
      rollout 侧 kernel 检查，GPU 显存/LoRA 覆盖率检查留给真机入口
      复用 B1 现成检查，不在这里重复）、`run_grpo_training`——veRL
      真实调用形态是 Hydra/YAML 驱动的 CLI（`python -m verl.trainer.
      main_ppo ...`），不是一个能在 Python 里直接构造调用的训练器
      类，本项目没有足够把握确认具体调用方式，显式
      `NotImplementedError`（同 B2 FSDP、B4 DPOTrainer 的处理方式）。
  - configs/train/grpo.yaml —— 验证过能被 `load_yaml_config` 正确
    加载；`lora_target_modules`沿用 B1/B2/B3/B4 同一份全32层覆盖
    列表；`harness_error_abort_threshold`直接写 0.01 并在注释里说明
    与 `HARNESS_ERROR_FLOOD_THRESHOLD` 保持同步，不是巧合。
  - pyproject.toml —— `train` extra 新增 `verl>=0.2`，mypy overrides
    新增 `verl.*`。
  - 单元测试 48 条（tests/unit/b5/：`compute_reward`对每个真实
    exec_code/comparison_result 组合穷举测过，`RolloutReward`/
    `RolloutGroup`两个构造函数级互斥约束单独测；`rollout_timing.py`
    用真实 SQLite db + 真实 `execute_many`测并发执行）、元测试 9 条
    （tests/meta/b5/：参数化覆盖全部四种 mask 场景真实产出 None 而非
    0.0；退化组检测在全同值/全 mask 两种场景真实触发，健康方差场景
    真实不触发；系统错占比超阈值真实中止训练，健康占比真实不中止）、
    烟测 1 条（tests/smoke/b5/：3题×k=4，全程走真实 sqlexec 并发
    执行+真实 A3 比对+真实 reward+真实组/步诊断，没有一处手写
    exec_code/comparison_result——和单测"构造 PredictionRecord 直接
    测单个函数"的风格刻意区分开）。`make verify-b5`（走泛用
    `verify-%`模式规则）三段全绿；mypy --strict 对 125 个源文件
    干净；全项目 926 个测试全绿，无跨轮 regressions；
    `check_no_cheating src`/`check_placeholders docs src` 均干净。

遇到什么问题：
  1. 写 `reward.py` 时对照 B4 `dpo.py` 的三档分类，发现 PLAN.md B5
    的奖励三档描述（结果匹配1.0/语法合法但结果错0.0/语法错0.0）和
    mask 清单（系统错+截断）都没提 UNDECIDABLE 该怎么处理——如果照
    字面意思归进"结果错→0.0"，等于对一个金标准数据本身就有缺陷的
    样本打了模型的 0 分，是 PLAN.md ★ 第1条点名的那类 bug 的又一个
    变种。选择显式 mask（DD-0036），和 B4 对同一情况的处理保持一致，
    而不是让两轮训练流程对"gold 执行失败"给出不同答案。
  2. `group_diagnostics.py`的退化判定最初写成"reward == 0.0 的
    数量占比"，意识到 PLAN.md 原文"组内全相同"这个更本质的表述——
    全 1.0 的组和全 0.0 的组在 GRPO advantage 计算里是同样白跑的
    （组内归一化，全同值 → 方差为0 → advantage 恒为0），改成
    `len(set(scored)) <= 1`的通用判定，测试里专门加了
    `test_all_one_is_also_degenerate`防止这个判断退化回"只看是否
    全 0"的特殊情况。
  3. veRL 真实的 Python 调用入口本项目查证后没有足够把握——它更像
    Hydra config-driven 的 CLI 工具而不是一个可以直接 import 构造的
    训练器类（这一点和 TRL 的 `DPOTrainer`、LLaMA-Factory 的
    `llamafactory-cli`都不一样，后两者至少有一种确定的接线方式）。
    没有编一个可能是错的 `import verl; verl.Trainer(...)`式调用，
    `run_grpo_training`对已安装 verl 的分支显式 `NotImplementedError`，
    真实调用形态留给真机验证。

怎么解决的：见上。

测出什么数字：无（无 GPU，本轮不产生任何真实 GRPO rollout/reward/
训练曲线数字）。tests/smoke/b5/ 里 3 题×k=4 的真实 sqlexec 执行+比对
只用来验证 reward→group诊断→step诊断这条链路本身接得通，不是任何
真实模型的 rollout 结果。

诚实清单：
  - 没做：任何真实的 veRL rollout/训练（本沙箱没有 verl、没有 vLLM、
    没有 GPU；`run_grpo_training`对已装 verl 的分支是
    `NotImplementedError`，真实调用形态未接线）；rollout 侧 vLLM 的
    GDN kernel 检测从未在真实 vLLM 实例上跑过（`guard_rollout_
    kernel_status`复用的 `detect_gdn_kernel_status`本身在本沙箱里
    唯一走到的分支是"无 CUDA 设备→degraded"，见 `tests/unit/b5/
    test_kernel_guard.py`的说明）；10 步烟测从未针对真实显存/真实
    流水线跑过——`GrpoSmokeCriteria`只验证过判定逻辑本身的正确性。
  - 假设了什么：`rollout_k=8`（配置文件里的示例值，PLAN.md 没有像
    B4 的 k=4 那样给 B5 一个字面数字，8 是发表过的 GRPO 方案里常见
    的组大小选择，未经本项目调参）；veRL 真实调用形态是 Hydra/YAML
    驱动的 CLI（本项目查证后的最佳理解，未在真实安装上验证）。
  - 已知局限：`compute_reward`把 UNSAFE_STATEMENT 归进"模型自己的
    错，reward=0.0"而不是单独 mask 或单独扣分——这是本轮基于"和 B4
    `dpo.py`处理同一 gap 的方式保持一致"这个理由做的选择，PLAN.md
    原文完全没提到这种候选，真机第一批真实 rollout 出现大量
    UNSAFE_STATEMENT 时（比如模型学坏了开始批量生成危险语句），
    reward 曲线会不会因此产生某种需要额外分析的模式，本轮无法验证。
```

```
夜间队列编排 night_queue.py（同一 session）—— PLAN.md 原文标注"单独
session"，本轮是 CLAUDECODEPROMPTS.md 全部 19 轮里的最后一轮，收尾
整个 A/B 两道工程
做了什么：
  - scripts/night_queue.py —— CLAUDE.md §6.2 八条要求逐条落地：
    - `build_night_queue`：稳定排序按 `TaskRiskLevel`（LOW/MEDIUM/
      HIGH）升序（项1），同风险任务保持调用方传入的相对顺序。
    - `run_task_with_watchdog`：任何真实异常都被捕获转成 `FAILED`
      outcome 而不是往外抛（项2"；语义不是&&语义"），队列循环因此
      永远能走到下一个任务；`KeyboardInterrupt`（真实 SIGINT，以及
      `_install_sigterm_as_keyboard_interrupt`把真实 SIGTERM 也转换
      成同一异常）单独识别成 `INTERRUPTED`，且中断后队列不再继续
      跑后续任务——因为是整个进程要退出，不是单个任务失败。
    - `write_task_manifest`：只给真正跑过的任务（COMPLETED/FAILED/
      INTERRUPTED，PLAN.md 项3 字面三态）写 manifest，`SMOKE_TEST_
      NOT_PASSED`/`DEFERRED`两种"从未执行"状态不写（DD-0037）。
    - watchdog 超时：`ThreadPoolExecutor`+`Future.result(timeout=…)`
      （项4，复用 A2 `sqlexec/pool.py`同款并发原语解决"别让一个任务
      卡住整条队列"），超时算 `FAILED`不是无声挂起。
    - `time_budget_s`+循环内"预算耗尽→标记 `DEFERRED`"（项5超配+
      顺延——真实队列在 `main()`里超配多少由调用方自己决定，这个
      模块负责的是"超出预算的部分不丢、诚实标记"这一半）。
    - `render_queue_snapshot`/`write_queue_snapshot`（项6，启动前
      打印完整队列+预估耗时，落盘 `artifacts/queues/<date>.json`）。
    - 烟测钩子 `NightTask.smoke_test: Callable[[], CheckStatus] | None`
      ——直接复用 A0 `CheckStatus`白名单（PASS/FAIL/SKIP），只有
      `PASS`才会真的调用 `task.run()`；`SMOKE_TEST_NOT_PASSED`这一个
      状态同时覆盖 FAIL 和 SKIP，两者从不等同于 `COMPLETED`（项7 ★
      "烟测跳过不得记为通过"，和 CLAUDE.md §1.5
      `test_skip_is_not_pass`是同一条规则的直接代码化）。
    - `write_heartbeat`：循环内按真实耗时每 `heartbeat_interval_s`
      （默认 900s=15分钟）写一次，循环结束后无论如何再写最后一次
      （项8——保证队列即使中途异常退出，磁盘上也有它最后的真实状态）。
  - scripts/night_queue_fixtures/example_tasks.py —— 真实、安全可跑
    的示例任务（pyproject.toml 里 mypy exclude 早在 Round 0 就已经
    预留了这个目录，本轮确认并填上内容）：两个 LOW 风险确定性任务
    直接 subprocess 调用本项目自己已有的 `check_no_cheating.py`/
    `check_placeholders.py`（真实、安全、CPU-only，是"确定性任务"
    的现成例子，不是编的占位任务）；一个 HIGH 风险占位任务演示
    "真实 GPU 依赖任务该长什么样"——调用 B5 `check_verl_available`
    真实探测，未装则诚实 `raise`，不是假装训练成功。
  - Makefile 的 `night-queue` 目标（`$(PY) scripts/night_queue.py`）
    在 Round 0 就已经存在，本轮第一次真正被填上内容并跑通。
  - 手动完整跑通一次 `make night-queue`：3 个任务（2 个真实确定性
    检查 COMPLETED + 1 个 HIGH 风险 GRPO 占位任务因为本沙箱没装 verl
    诚实 FAILED），产出真实 `artifacts/queues/20260824.json`队列
    快照、`artifacts/queues/20260824-heartbeat.json`心跳、
    `artifacts/queues/20260824/manifests/`下 3 份真实每任务
    manifest——不是构造的示例数据。
  - 单元测试 20 条（tests/unit/night_queue/：`NightTask`验证、风险
    排序稳定性、watchdog 的六种真实分支——健康完成/真实异常/
    KeyboardInterrupt/烟测PASS/烟测FAIL/烟测SKIP/超时——队列快照
    写入读回、manifest 只给三态写的边界情况）、元测试 2 条
    （tests/meta/night_queue/：本轮自己的验收标准直接写成元测试——
    队列中间一个必然失败的任务真实不阻断后续、且五个任务状态全部
    正确记录；全健康队列证明机制不是永远卡在第一个失败上）、烟测 1
    条（tests/smoke/night_queue/：真实超配队列+时间预算耗尽触发真实
    DEFERRED顺延+真实烟测拦截，一次跑通验证风险排序/预算顺延/烟测
    跳过/manifest 边界四件事）。`make verify-night_queue`（走泛用
    `verify-%`模式规则，无需新增 Makefile 目标）三段全绿；
    mypy --strict 对 `src/modelhub`125 个源文件干净，`scripts/
    night_queue.py`/`example_tasks.py`单独跑 mypy 也干净（不在
    CI 强制范围内，但保持同等质量标准）；全项目 949 个测试全绿，
    无跨轮 regressions；`check_no_cheating src`/`check_placeholders
    docs src`均干净，`check_no_cheating scripts`额外确认本轮新增
    文件没有引入新 finding（scripts/ 下现有 5 条历史 finding 均与
    本轮无关）。

遇到什么问题：
  1. `scripts/night_queue_fixtures/example_tasks.py`最初用
    `from scripts.night_queue import ...`这种包限定路径导入，
    mypy 报"Source file found twice under different module names"——
    查了一下才发现本项目 `tests/conftest.py`早就把 `scripts/`目录
    本身加进了 `sys.path`（供 `demo.py`/`bad_model_drill.py`这类
    脚本被测试用裸模块名 `import demo`的方式导入），意味着
    `scripts/`内部互相 import 也应该用裸模块名（`from night_queue
    import ...`），不是当成一个真正的 `scripts.xxx`包。改成裸导入 +
    给 `night_queue_fixtures/`补一个 `__init__.py`（让它自己能被
    当包导入）后 mypy 干净。
  2. 设计"watchdog：任务死亡后立即启动下一个"这条要求时，意识到本
    项目里所有任务函数目前都是同步阻塞的 Python 可调用对象（不是
    真的独立子进程/守护进程），"任务死亡"最贴近的真实含义是"这个
    可调用对象抛出异常（包括真实子进程 `subprocess.run(...,
    check=True)`崩溃时会转换成的 `CalledProcessError`）"——没有为了
    显得"更像真实 watchdog"去引入一层不存在的进程级探活机制，而是
    把"watchdog"的真实价值落在两处能验证的地方：循环本身不插入任何
    人为延迟（下一个任务立即开始），以及一个可选的
    `watchdog_timeout_s`安全网（`ThreadPoolExecutor`+
    `Future.result(timeout=…)`）防止某个任务因为自己没写好内部
    超时而挂住整条队列。
  3. 每任务 manifest 该给哪些状态写，PLAN.md 项3 字面只列了三态
    （COMPLETED/FAILED/INTERRUPTED），但 `QueueTaskStatus`为了实现
    项5（超配顺延）和项7（烟测未过）额外多了 `DEFERRED`/
    `SMOKE_TEST_NOT_PASSED`两个状态。写完 `write_task_manifest`
    第一版时意识到如果对这两种"根本没跑起来"的状态也生成 manifest，
    要么留一堆没意义的 `None`字段，要么就是在编造这次执行本不存在
    的"起止时间"——改成只对真正跑过的三态写 manifest，另外两种状态
    只在队列级心跳/快照里出现（DD-0037）。

怎么解决的：见上。

测出什么数字：`artifacts/queues/20260824.json`/`20260824-heartbeat.
json`/`20260824/manifests/*.json`——一次真实 `make night-queue`
跑出的队列编排产物（2 个真实反作弊/占位符检查 COMPLETED + 1 个 GRPO
占位任务因本沙箱无 verl 诚实 FAILED）。这些是队列编排机制本身跑通的
证据，不是任何训练/评估性能数字。

诚实清单：
  - 没做：真实的多小时超配夜间队列（本轮手动跑的示例队列只有 3 个
    任务、几秒钟跑完，`time_budget_s=8*3600`从未真正被真实任务序列
    填满过）；`watchdog_timeout_s`安全网从未在真实会挂起的任务上
    触发过（单测里用 `time.sleep`模拟，不是真实训练/推理任务卡死的
    场景）；真实 SIGTERM 信号从未在真实运行的 `make night-queue`
    进程上发送/验证过（`_install_sigterm_as_keyboard_interrupt`的
    信号处理器逻辑单测走的是直接 `raise KeyboardInterrupt`模拟，
    不是真实操作系统信号投递）。
  - 假设了什么：`example_tasks.py`里三个示例任务的
    `estimated_duration_s`（30s/15s/8小时）是说明性占位数字，不是
    任何真实测量；HIGH 风险 GRPO 占位任务的 `watchdog_timeout_s=60`
    同样是示例值。
  - 已知局限：这是 CLAUDECODEPROMPTS.md 全部 19 轮（Round0 + A0-A12
    + B1-B5 + night_queue）里的最后一轮——整个项目从这里开始进入
    "代码全部写完、等待真机验证"的状态，`docs/build-log.md`里累计
    的"没做什么"清单（GPU 训练、真实模型推理、真实 rollout 等）在
    真机拿到之前都不会自动解决，需要按 CLAUDE.md §0"所有对外声称的
    数字必须能追溯到一次真实 run 的落盘产物"这条铁律，逐条真机复核
    后才能对外引用。
```
