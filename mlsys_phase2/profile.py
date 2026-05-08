from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .utils import discover_cases, require_executable, run_command, validate_case_dirs

DEFAULT_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

RUNNER = r'''
import argparse
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def load_inputs(base_dir: str):
    base = Path(base_dir)
    W = torch.load(base / "W.pt", map_location="cpu").contiguous().cuda()
    X = torch.load(base / "X.pt", map_location="cpu").contiguous().cuda()
    A = torch.load(base / "A.pt", map_location="cpu").contiguous().cuda()
    B = torch.load(base / "B.pt", map_location="cpu").contiguous().cuda()
    return W, X, A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cu-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()
    module = load(
        name="optimized_lora_profile_ext",
        sources=[args.cu_path],
        verbose=False,
        build_directory=args.build_dir,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    W, X, A, B = load_inputs(args.input_dir)
    with torch.no_grad():
        for _ in range(3):
            _ = module.forward(W, X, A, B)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            _ = module.forward(W, X, A, B)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
'''


def profile_lora_code(
    code: str,
    inputs_root: str | None = None,
    metrics: list[str] | None = None,
    iters: int = 8,
    timeout: int = 300,
) -> dict[str, Any]:
    cases = discover_cases(inputs_root)
    missing = validate_case_dirs(inputs_root)
    if missing:
        return {
            "ok": False,
            "cases": [{"d": c.d, "ok": False, "error": "missing input files", "stdout": "", "stderr": ""} for c in cases],
            "error": "缺少输入文件: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        }

    ncu = require_executable("ncu")
    if not ncu:
        return {
            "ok": False,
            "cases": [{"d": c.d, "ok": False, "error": "找不到 ncu 可执行文件", "stdout": "", "stderr": ""} for c in cases],
            "error": "找不到 ncu，请安装 NVIDIA Nsight Compute 并确保 ncu 在 PATH 中。",
        }

    selected_metrics = metrics or DEFAULT_NCU_METRICS
    with tempfile.TemporaryDirectory(prefix="mlsys_lora_ncu_") as tmp:
        tmp_path = Path(tmp)
        cu_path = tmp_path / "optimized_lora.cu"
        runner_path = tmp_path / "profile_runner.py"
        cu_path.write_text(code, encoding="utf-8")
        runner_path.write_text(RUNNER, encoding="utf-8")
        results: list[dict[str, Any]] = []

        for case in cases:
            build_dir = tmp_path / f"build_d{case.d}"
            build_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                ncu,
                "--target-processes",
                "all",
                "--metrics",
                ",".join(selected_metrics),
                "--csv",
                "--page",
                "raw",
                "python",
                str(runner_path),
                "--cu-path",
                str(cu_path),
                "--input-dir",
                str(case.base_dir),
                "--build-dir",
                str(build_dir),
                "--iters",
                str(iters),
            ]
            try:
                proc = run_command(cmd, cwd=tmp_path, timeout=timeout)
                ok = proc.returncode == 0
                results.append(
                    {
                        "d": case.d,
                        "ok": ok,
                        "returncode": proc.returncode,
                        "metrics": selected_metrics,
                        "stdout": proc.stdout[-20000:],
                        "stderr": proc.stderr[-20000:],
                        "error": None if ok else "ncu 执行失败",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "d": case.d,
                        "ok": False,
                        "returncode": None,
                        "metrics": selected_metrics,
                        "stdout": "",
                        "stderr": "",
                        "error": str(exc),
                    }
                )

    ok = all(r["ok"] for r in results)
    return {"ok": ok, "cases": results, "error": None if ok else "存在失败的 ncu profile case"}
