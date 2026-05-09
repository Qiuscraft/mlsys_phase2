from __future__ import annotations

import csv
import io
import re
import tempfile
import random
from pathlib import Path
from typing import Any

from .utils import discover_cases, require_executable, run_command

DEFAULT_NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
]

MAX_NCU_KERNELS_IN_SUMMARY = 20

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


def _maybe_float(value: str) -> float | None:
    text = value.strip().strip('"').replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _looks_like_kernel_name(value: str) -> bool:
    text = value.strip().strip('"')
    if not text or len(text) > 500:
        return False
    lowered = text.lower()
    if lowered in {"kernel name", "name", "python", "python3", "python3.10"}:
        return False
    return (
        "sgemm" in lowered
        or "gemm" in lowered
        or "cutlass" in lowered
        or text.startswith("void ")
        or "::" in text
        or "<<<" in text
    )


def _summarize_metric_values(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    if all(value == 0.0 for value in values):
        return None
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "values": values[:MAX_NCU_KERNELS_IN_SUMMARY],
    }


def summarize_ncu_stdout(stdout: str, selected_metrics: list[str]) -> dict[str, Any]:
    """Return a compact ncu summary for LLM prompts.

    Nsight Compute CSV formats differ across versions/pages. This parser is
    intentionally defensive: it extracts kernel-like names from any CSV field,
    and extracts selected metric values either from CSV header columns or from
    rows that contain the metric name. The full raw stdout is deliberately not
    returned, because it can be very large and can make LLM requests time out.
    """
    rows: list[list[str]] = []
    try:
        rows = [row for row in csv.reader(io.StringIO(stdout)) if row]
    except csv.Error:
        rows = []

    kernel_counts: dict[str, int] = {}
    metric_values: dict[str, list[float]] = {metric: [] for metric in selected_metrics}

    metric_column_indexes: dict[str, list[int]] = {metric: [] for metric in selected_metrics}
    if rows:
        for idx, cell in enumerate(rows[0]):
            normalized = cell.strip().strip('"')
            for metric in selected_metrics:
                if normalized == metric or metric in normalized:
                    metric_column_indexes[metric].append(idx)

    for row in rows:
        for cell in row:
            if _looks_like_kernel_name(cell):
                name = cell.strip().strip('"')
                kernel_counts[name] = kernel_counts.get(name, 0) + 1

        for metric, indexes in metric_column_indexes.items():
            for idx in indexes:
                if idx < len(row):
                    value = _maybe_float(row[idx])
                    if value is not None:
                        metric_values[metric].append(value)

        row_text = ",".join(row)
        for metric in selected_metrics:
            if metric not in row_text:
                continue
            for idx, cell in enumerate(row):
                if metric not in cell:
                    continue
                candidate_cells = row[idx + 1 : idx + 8] + row[max(0, idx - 3) : idx]
                for candidate in candidate_cells:
                    value = _maybe_float(candidate)
                    if value is not None:
                        metric_values[metric].append(value)
                        break

    # Fallback for non-CSV ncu output such as table/details pages. Avoid
    # double-counting kernel names when CSV parsing already found them.
    use_text_kernel_fallback = not kernel_counts
    for line in stdout.splitlines():
        if use_text_kernel_fallback:
            for match in re.finditer(r"(?:void\s+)?[A-Za-z_][\w:<>~.,\s*&()]+(?:sgemm|gemm|cutlass)[\w:<>~.,\s*&()]*", line, re.IGNORECASE):
                name = " ".join(match.group(0).split()).strip('" ,')
                if _looks_like_kernel_name(name):
                    kernel_counts[name] = kernel_counts.get(name, 0) + 1
        for metric in selected_metrics:
            if metric not in line:
                continue
            tail = line.split(metric, 1)[1]
            match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tail)
            if match:
                value = _maybe_float(match.group(0))
                if value is not None:
                    metric_values[metric].append(value)

    kernels = [
        {"name": name, "count": count}
        for name, count in sorted(kernel_counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_NCU_KERNELS_IN_SUMMARY]
    ]
    metrics: dict[str, Any] = {}
    for metric, values in metric_values.items():
        summary = _summarize_metric_values(values)
        if summary is not None:
            metrics[metric] = summary
    return {
        "kernel_count": sum(kernel_counts.values()),
        "kernels": kernels,
        "metrics": metrics,
        "raw_stdout_chars": len(stdout),
    }


def profile_lora_code(
    code: str,
    inputs_root: str | None = None,
    metrics: list[str] | None = None,
    iters: int = 8,
    timeout: int = 300,
) -> dict[str, Any]:
    cases = discover_cases(inputs_root)
    if cases:
        cases = [random.choice(cases)]

    missing: list[str] = []
    for case in cases:
        for name in ("W.pt", "X.pt", "A.pt", "B.pt"):
            p = case.base_dir / name
            if not p.exists():
                missing.append(str(p))
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
                        "requested_metrics": selected_metrics,
                        "summary": summarize_ncu_stdout(proc.stdout, selected_metrics),
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
                        "requested_metrics": selected_metrics,
                        "summary": {"kernel_count": 0, "kernels": [], "metrics": {}, "raw_stdout_chars": 0},
                        "stdout": "",
                        "stderr": "",
                        "error": str(exc),
                    }
                )

    ok = all(r["ok"] for r in results)
    return {
        "ok": ok,
        "cases": results,
        "sampled_d": results[0]["d"] if results else None,
        "error": None if ok else "存在失败的 ncu profile case",
    }
