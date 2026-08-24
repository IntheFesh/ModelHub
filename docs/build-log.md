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
