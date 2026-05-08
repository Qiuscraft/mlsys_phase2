from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .bench import benchmark_lora_code
from .profile import profile_lora_code
from .prompts import INITIAL_USER_PROMPT, SYSTEM_PROMPT, optimize_prompt, repair_prompt
from .utils import dumps_result, project_root, strip_markdown_code_fence


def _openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - depends on installed package
        raise RuntimeError("需要安装 openai 包才能运行 Agent。") from exc
    kwargs: dict[str, str] = {}
    if os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def call_llm(user_prompt: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
    )
    content = resp.choices[0].message.content or ""
    return strip_markdown_code_fence(content)


def candidate_is_better(result: dict[str, Any], best_speedup: float) -> bool:
    return bool(result.get("ok")) and float(result.get("average_speedup") or 0.0) > best_speedup


def write_best(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def run_agent(
    inputs_root: str | None = None,
    output_path: str | None = None,
    max_init_attempts: int = 8,
    max_opt_iters: int | None = None,
    warmup: int = 10,
    iters: int = 50,
    profile_iters: int = 8,
) -> int:
    out = Path(output_path).expanduser().resolve() if output_path else project_root() / "optimized.cu"
    opt_iters = max_opt_iters if max_opt_iters is not None else int(os.getenv("MAX_OPT_ITERS", "10"))

    print("生成初始 CUDA 代码...", flush=True)
    code = call_llm(INITIAL_USER_PROMPT)
    best_result: dict[str, Any] | None = None
    best_speedup = 0.0

    for attempt in range(1, max_init_attempts + 1):
        print(f"初始正确性/性能检查 attempt={attempt}/{max_init_attempts}", flush=True)
        result = benchmark_lora_code(code, inputs_root=inputs_root, warmup=warmup, iters=iters)
        print(dumps_result(result), flush=True)
        if result.get("ok"):
            best_result = result
            best_speedup = float(result.get("average_speedup") or 0.0)
            write_best(out, code)
            print(f"初始代码已采纳: {out}, average_speedup={best_speedup:.6f}", flush=True)
            break
        code = call_llm(repair_prompt(code, dumps_result(result)))

    if best_result is None:
        print("无法生成通过正确性检查的初始代码。", file=sys.stderr, flush=True)
        return 1

    for step in range(1, opt_iters + 1):
        print(f"优化迭代 {step}/{opt_iters}: ncu profile", flush=True)
        profile_result = profile_lora_code(code, inputs_root=inputs_root, iters=profile_iters)
        print(dumps_result(profile_result), flush=True)

        print(f"优化迭代 {step}/{opt_iters}: 生成候选代码", flush=True)
        candidate = call_llm(optimize_prompt(code, dumps_result(best_result), dumps_result(profile_result), best_speedup))
        print(f"优化迭代 {step}/{opt_iters}: benchmark 候选代码", flush=True)
        result = benchmark_lora_code(candidate, inputs_root=inputs_root, warmup=warmup, iters=iters)
        print(dumps_result(result), flush=True)

        if candidate_is_better(result, best_speedup):
            code = candidate
            best_result = result
            best_speedup = float(result.get("average_speedup") or 0.0)
            write_best(out, code)
            print(f"采纳新代码: average_speedup={best_speedup:.6f}", flush=True)
        else:
            print(f"不采纳候选代码，保持 best average_speedup={best_speedup:.6f}", flush=True)

    print(f"完成。最佳代码: {out}, best average_speedup={best_speedup:.6f}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LoRA CUDA auto-optimization agent.")
    parser.add_argument("--inputs-root", default=None, help="默认使用项目根目录 inputs/")
    parser.add_argument("--output", default=None, help="默认写入项目根目录 optimized.cu")
    parser.add_argument("--max-init-attempts", type=int, default=8)
    parser.add_argument("--max-opt-iters", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile-iters", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(
        run_agent(
            inputs_root=args.inputs_root,
            output_path=args.output,
            max_init_attempts=args.max_init_attempts,
            max_opt_iters=args.max_opt_iters,
            warmup=args.warmup,
            iters=args.iters,
            profile_iters=args.profile_iters,
        )
    )


if __name__ == "__main__":
    main()
