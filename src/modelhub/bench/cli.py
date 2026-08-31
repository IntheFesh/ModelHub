"""`make bench PROFILE=<name>` / `python -m modelhub.bench.cli --profile <name>`.

This entrypoint was referenced by the Makefile's `bench` target since A8
but never actually existed as a module — `make bench` would crash with
`ModuleNotFoundError` instead of the graceful GPU-exclusivity rejection
CLAUDE.md §5.2 and this project's own docs describe. This fixes that gap.

CLAUDE.md §5.2: "每个 bench run 启动前用 nvidia-smi 检查目标 GPU 有无
其他进程，有则拒绝启动并报错." This module's first action — and, in any
sandbox without a real GPU, its *only* real action — is exactly that
check (`bench/gpu_guard.py::guard_gpu_exclusivity`, already implemented
and unit-tested). Everything past it (connecting to a live served model
endpoint, running `bench/sweep.py::run_sweep` against real traffic,
writing a manifest) genuinely needs a real GPU machine with a model
actually being served, which this environment has never had. Rather
than guess an unverified wiring shape for that part, this raises
`NotImplementedError` describing the precise next step — the same
discipline `train/dpo.py::run_dpo_training` and
`train/grpo/runner.py::run_grpo_training` already apply: 不确定就留
NotImplementedError, 不要在没有真实环境验证的情况下编一个大概率不对的
调用形状.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modelhub.bench.gpu_guard import guard_gpu_exclusivity
from modelhub.common.config import load_yaml_config
from modelhub.common.errors import ModelHubError, Stage
from modelhub.serve.model_profile import ModelProfile

_MODEL_PROFILES_DIR = Path("configs/serve/model_profiles")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="model profile name under configs/serve/model_profiles/ (e.g. arctic_7b, qwen3_5_9b)",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args(argv)

    profile_path = _MODEL_PROFILES_DIR / f"{args.profile}.yaml"
    if not profile_path.is_file():
        print(f"bench: no such model profile {profile_path}", file=sys.stderr)
        return 1

    try:
        guard_gpu_exclusivity(args.gpu_index)
    except ModelHubError as e:
        # the real error's full detail is printed above, never swallowed —
        # `return 1` below is a CLI process exit code, not a disguised
        # fake success value (same shape as check_no_cheating.py's main()).
        print(f"bench: {e}", file=sys.stderr)
        # check-no-cheating: allow=EXCEPT_RETURN_CONSTANT reason=CLI exit code, error printed above
        return 1

    profile, _config_hash = load_yaml_config(ModelProfile, profile_path, stage=Stage.SERVE)
    raise NotImplementedError(
        f"GPU {args.gpu_index} exclusivity confirmed for profile {profile.model_id!r} — but "
        f"connecting to a real served model endpoint and running bench/sweep.py::run_sweep "
        f"against it needs a real GPU machine with vLLM actually serving this model, which "
        f"this environment does not have. Wire this up against the real serve/ HTTP client "
        f"on first real GPU run rather than guessing its exact call shape untested "
        f"(CLAUDE.md: 不确定就留 NotImplementedError)."
    )


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
