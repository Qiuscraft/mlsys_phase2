from __future__ import annotations

import logging
import sys
from typing import Any

from .bench import benchmark_lora_code
from .logging_utils import setup_logging
from .profile import profile_lora_code
from .utils import strip_markdown_code_fence


logger = logging.getLogger(__name__)


def _summarize_result(result: dict[str, Any]) -> str:
    summary: dict[str, Any] = {
        "ok": result.get("ok"),
        "error": result.get("error"),
    }
    if "average_speedup" in result:
        summary["average_speedup"] = result.get("average_speedup")
    if "cases" in result:
        summary["case_count"] = len(result.get("cases") or [])
    return str(summary)


def _create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - depends on installed package
        raise RuntimeError("需要安装 mcp[cli] 才能启动 FastMCP server。") from exc

    mcp = FastMCP("mlsys-lora-agent")

    @mcp.tool()
    def benchmark_lora(code: str, inputs_root: str | None = None, warmup: int = 10, iters: int = 50) -> dict[str, Any]:
        """Compile and benchmark a single-file CUDA implementation for the LoRA operator."""
        logger.info(
            "模型调用 MCP 服务: benchmark_lora inputs_root=%s warmup=%s iters=%s code_chars=%s",
            inputs_root,
            warmup,
            iters,
            len(code),
        )
        result = benchmark_lora_code(strip_markdown_code_fence(code), inputs_root=inputs_root, warmup=warmup, iters=iters)
        logger.info("MCP 服务返回: benchmark_lora %s", _summarize_result(result))
        return result

    @mcp.tool()
    def profile_lora_ncu(code: str, inputs_root: str | None = None, iters: int = 8) -> dict[str, Any]:
        """Run NVIDIA Nsight Compute profiling for the LoRA CUDA implementation."""
        logger.info(
            "模型调用 MCP 服务: profile_lora_ncu inputs_root=%s iters=%s code_chars=%s",
            inputs_root,
            iters,
            len(code),
        )
        result = profile_lora_code(strip_markdown_code_fence(code), inputs_root=inputs_root, iters=iters)
        logger.info("MCP 服务返回: profile_lora_ncu %s", _summarize_result(result))
        return result

    return mcp


def main() -> None:
    setup_logging(log_to_file=True, console_stream=sys.stderr)
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
