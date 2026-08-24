.PHONY: help venv install lint typecheck test verify-% verify-a11-% smoke bench \
        check-cheating check-placeholders check-pollution render-docs demo night-queue clean

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

help:
	@echo "modelhub Makefile"
	@echo "  make venv               创建虚拟环境"
	@echo "  make install            安装依赖 (dev extras)"
	@echo "  make lint               ruff check + format --check"
	@echo "  make typecheck          mypy --strict src/"
	@echo "  make test               全部 pytest（不含 requires_gpu/requires_network）"
	@echo "  make verify-<id>        单轮验收，例如 make verify-a0"
	@echo "  make check-cheating     AST 反作弊静态扫描"
	@echo "  make check-placeholders 占位符扫描，必须为 0"
	@echo "  make check-pollution    校验 artifacts/runs 下每个 run 的三个污染位"
	@echo "  make render-docs        从 artifacts 重新生成 docs/ 下的数字页面"
	@echo "  make demo               起服务→发请求→展示监控→触发门禁拦截→展示incident-log"
	@echo "  make smoke PROFILE=x    烟测"
	@echo "  make bench PROFILE=x    压测（需真实 GPU 环境，本沙箱会拒绝启动）"
	@echo "  make night-queue        跑夜间任务队列"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev,data,serve,gateway,monitor,sqlexec]"

lint:
	$(RUFF) check src tests scripts
	$(RUFF) format --check src tests scripts

fmt:
	$(RUFF) check --fix src tests scripts
	$(RUFF) format src tests scripts

typecheck:
	$(MYPY) src/modelhub

test:
	$(PYTEST) -m "not requires_gpu and not requires_network" -v

check-cheating:
	$(PY) scripts/check_no_cheating.py src

check-placeholders:
	$(PY) scripts/check_placeholders.py docs src

check-pollution:
	$(PY) scripts/audit_pollution.py

render-docs:
	$(PY) scripts/render_docs.py

# make verify-a0, make verify-a1, ... make verify-b5
# 约定：每轮的单元测试放 tests/unit/<id>/，元测试放 tests/meta/<id>/，
# 烟测放 tests/smoke/<id>/。verify 目标三者都跑（存在才跑，不存在跳过并打印）。
# ★ -m "not requires_gpu and not requires_network" 是本沙箱环境的显式豁免，
#   不是"跳过=通过"：pytest 会把被排除的用例打印为 deselected，
#   在真实 GPU/联网机器上应去掉这个过滤器跑全量。
VERIFY_MARKEXPR := not requires_gpu and not requires_network
verify-%:
	@echo "════════════════════════════════════════════════════════════"
	@echo " make verify-$*"
	@echo "════════════════════════════════════════════════════════════"
	@$(RUFF) check src/modelhub tests/unit/$* tests/meta/$* 2>/dev/null || $(RUFF) check src/modelhub
	@if [ -d tests/unit/$* ]; then \
		echo "-- unit --"; $(PYTEST) tests/unit/$* -v -m "$(VERIFY_MARKEXPR)"; \
	else echo "-- unit: 无 tests/unit/$* --"; fi
	@if [ -d tests/meta/$* ]; then \
		echo "-- meta（证明检查能红） --"; $(PYTEST) tests/meta/$* -v -m "$(VERIFY_MARKEXPR)"; \
	else echo "-- meta: 无 tests/meta/$* --"; fi
	@if [ -d tests/smoke/$* ]; then \
		echo "-- smoke --"; $(PYTEST) tests/smoke/$* -v -m "$(VERIFY_MARKEXPR)"; \
	else echo "-- smoke: 无 tests/smoke/$* --"; fi

# A11 has six independent experiment groups (bench/experiments/) — PLAN.md's
# "五组实验...每组独立" plus a v2-added sixth (MFU/MBU) — each verifiable on
# its own via make verify-a11-<n>. These are explicit targets so they take
# precedence over the verify-% pattern above; verify-a11 (via that pattern)
# still runs the whole tests/{unit,meta,smoke}/a11/ suite together.
A11_GROUP_1 := quantization
A11_GROUP_2 := prefix_cache
A11_GROUP_3 := speculative_decoding
A11_GROUP_4 := constrained_decoding
A11_GROUP_5 := engine_comparison
A11_GROUP_6 := mfu_mbu_comparison

verify-a11-%:
	@echo "════════════════════════════════════════════════════════════"
	@echo " make verify-a11-$* ($(A11_GROUP_$*))"
	@echo "════════════════════════════════════════════════════════════"
	@$(RUFF) check src/modelhub/bench/experiments/$(A11_GROUP_$*).py tests/unit/a11/test_$(A11_GROUP_$*).py
	@echo "-- unit --"
	@$(PYTEST) tests/unit/a11/test_$(A11_GROUP_$*).py -v -m "$(VERIFY_MARKEXPR)"

smoke:
	$(PYTEST) tests/smoke/$(PROFILE) -v -m "not requires_gpu and not requires_network"

bench:
	$(PY) -m modelhub.bench.cli --profile $(PROFILE)

demo:
	$(PY) scripts/demo.py

night-queue:
	$(PY) scripts/night_queue.py

clean:
	rm -rf $(VENV) .mypy_cache .ruff_cache .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
