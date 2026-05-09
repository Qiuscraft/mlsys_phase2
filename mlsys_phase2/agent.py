from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from .bench import benchmark_lora_code
from .logging_utils import setup_logging
from .profile import profile_lora_code
from .prompts import INITIAL_USER_PROMPT, SYSTEM_PROMPT, optimize_prompt, repair_prompt
from .utils import dumps_result, project_root, strip_markdown_code_fence


logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load project-level .env for the agent without overriding exported variables."""
    try:
        from dotenv import load_dotenv
    except Exception as exc:  # pragma: no cover - depends on installed package
        raise RuntimeError("需要安装 python-dotenv 包才能从 .env 读取环境变量。") from exc

    load_dotenv(project_root() / ".env", override=False)


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


def log_tool_result(tool_name: str, result: dict[str, Any]) -> None:
    summary: dict[str, Any] = {
        "ok": result.get("ok"),
        "error": result.get("error"),
    }
    if "average_speedup" in result:
        summary["average_speedup"] = result.get("average_speedup")
    if "cases" in result:
        summary["case_count"] = len(result.get("cases") or [])
    log_fn = logger.info if result.get("ok") else logger.error
    log_fn("返回 %s: %s", tool_name, dumps_result(summary))


def run_agent(
    inputs_root: str | None = None,
    output_path: str | None = None,
    max_init_attempts: int = 8,
    max_opt_iters: int | None = None,
    warmup: int = 10,
    iters: int = 50,
    profile_iters: int = 8,
) -> int:
    log_path = setup_logging(log_to_file=True)
    logger.info("日志文件: %s", log_path)
    load_env()

    out = Path(output_path).expanduser().resolve() if output_path else project_root() / "optimized.cu"
    opt_iters = max_opt_iters if max_opt_iters is not None else int(os.getenv("MAX_OPT_ITERS", "10"))

    logger.info("初始 CUDA System Prompt:\n%s", SYSTEM_PROMPT)
    logger.info("初始 CUDA User Prompt:\n%s", INITIAL_USER_PROMPT)
    logger.info("生成初始 CUDA 代码...")
    code = call_llm(INITIAL_USER_PROMPT)
    best_result: dict[str, Any] | None = None
    best_speedup = 0.0

    for attempt in range(1, max_init_attempts + 1):
        logger.info("初始正确性/性能检查 attempt=%s/%s", attempt, max_init_attempts)
        logger.info(
            "进入 benchmark: phase=initial attempt=%s/%s inputs_root=%s warmup=%s iters=%s code_chars=%s\n%s",
            attempt,
            max_init_attempts,
            inputs_root,
            warmup,
            iters,
            len(code),
            code,
        )
        result = benchmark_lora_code(code, inputs_root=inputs_root, warmup=warmup, iters=iters)
        log_tool_result("benchmark", result)
        logger.info("%s", dumps_result(result))
        if result.get("ok"):
            best_result = result
            best_speedup = float(result.get("average_speedup") or 0.0)
            write_best(out, code)
            logger.info("初始代码已采纳: %s, average_speedup=%.6f", out, best_speedup)
            break
        code = call_llm(repair_prompt(code, dumps_result(result)))

    if best_result is None:
        logger.error("无法生成通过正确性检查的初始代码。")
        return 1

    for step in range(1, opt_iters + 1):
        logger.info("优化迭代 %s/%s: ncu profile", step, opt_iters)
        logger.info(
            "进入 ncu: phase=optimize step=%s/%s inputs_root=%s iters=%s code_chars=%s\n%s",
            step,
            opt_iters,
            inputs_root,
            profile_iters,
            len(code),
            code,
        )
        profile_result = profile_lora_code(code, inputs_root=inputs_root, iters=profile_iters)
        log_tool_result("ncu", profile_result)
        logger.info("%s", dumps_result(profile_result))

        logger.info("优化迭代 %s/%s: 生成候选代码", step, opt_iters)
        candidate = call_llm(optimize_prompt(code, dumps_result(best_result), dumps_result(profile_result), best_speedup))
        logger.info("优化迭代 %s/%s: benchmark 候选代码", step, opt_iters)
        logger.info(
            "进入 benchmark: phase=candidate step=%s/%s inputs_root=%s warmup=%s iters=%s code_chars=%s\n%s",
            step,
            opt_iters,
            inputs_root,
            warmup,
            iters,
            len(candidate),
            candidate,
        )
        result = benchmark_lora_code(candidate, inputs_root=inputs_root, warmup=warmup, iters=iters)
        log_tool_result("benchmark", result)
        logger.info("%s", dumps_result(result))

        if candidate_is_better(result, best_speedup):
            code = candidate
            best_result = result
            best_speedup = float(result.get("average_speedup") or 0.0)
            write_best(out, code)
            logger.info("采纳新代码: average_speedup=%.6f", best_speedup)
        else:
            logger.warning("不采纳候选代码，保持 best average_speedup=%.6f", best_speedup)

    logger.info("完成。最佳代码: %s, best average_speedup=%.6f", out, best_speedup)
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
