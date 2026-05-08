from __future__ import annotations

from typing import Any

from .bench import benchmark_lora_code
from .profile import profile_lora_code
from .utils import strip_markdown_code_fence


def _create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - depends on installed package
        raise RuntimeError("需要安装 mcp[cli] 才能启动 FastMCP server。") from exc

    mcp = FastMCP("mlsys-lora-agent")

    @mcp.tool()
    def benchmark_lora(code: str, inputs_root: str | None = None, warmup: int = 10, iters: int = 50) -> dict[str, Any]:
        """Compile and benchmark a single-file CUDA implementation for the LoRA operator."""
        return benchmark_lora_code(strip_markdown_code_fence(code), inputs_root=inputs_root, warmup=warmup, iters=iters)

    @mcp.tool()
    def profile_lora_ncu(code: str, inputs_root: str | None = None, iters: int = 8) -> dict[str, Any]:
        """Run NVIDIA Nsight Compute profiling for the LoRA CUDA implementation."""
        return profile_lora_code(strip_markdown_code_fence(code), inputs_root=inputs_root, iters=iters)

    return mcp


def main() -> None:
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
