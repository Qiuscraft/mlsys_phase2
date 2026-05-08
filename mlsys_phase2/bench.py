from __future__ import annotations

import hashlib
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

from .utils import discover_cases, validate_case_dirs


def _import_torch():
    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError("需要安装带 CUDA 支持的 PyTorch 才能运行 benchmark。") from exc
    return torch, load


def load_inputs(base_dir: Path):
    torch, _ = _import_torch()
    tensors = []
    for name in ("W.pt", "X.pt", "A.pt", "B.pt"):
        tensors.append(torch.load(base_dir / name, map_location="cpu").contiguous().cuda())
    W, X, A, B = tensors
    return W, X, A, B


def reference_impl(W, X, A, B):
    torch, _ = _import_torch()
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def build_module(cu_path: Path, build_dir: Path):
    _, load = _import_torch()
    digest = hashlib.sha1(cu_path.read_bytes()).hexdigest()[:12]
    module = load(
        name=f"optimized_lora_ext_{digest}",
        sources=[str(cu_path)],
        verbose=False,
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    return module


def check_correctness(y, y_ref) -> tuple[bool, float, float]:
    torch, _ = _import_torch()
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    return bool(passed), float(max_abs_err), float(rel_l2_err)


def benchmark_fn(fn: Callable, W, X, A, B, warmup: int = 10, iters: int = 50) -> float:
    torch, _ = _import_torch()
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(W, X, A, B)
        torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn(W, X, A, B)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    times.sort()
    return times[len(times) // 2]


def _empty_case_result(d: int, error: str | None = None) -> dict[str, Any]:
    return {
        "d": d,
        "correct": False,
        "max_abs_err": None,
        "rel_l2_err": None,
        "student_median_ms": None,
        "torch_median_ms": None,
        "speedup": 0.0,
        "error": error,
    }


def benchmark_lora_code(
    code: str,
    inputs_root: str | None = None,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    """Compile a single-file CUDA extension and benchmark it on all LoRA cases."""
    cases = discover_cases(inputs_root)
    missing = validate_case_dirs(inputs_root)
    if missing:
        return {
            "ok": False,
            "average_speedup": 0.0,
            "cases": [_empty_case_result(c.d, "missing input files") for c in cases],
            "error": "缺少输入文件: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        }

    try:
        torch, _ = _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() 为 False，无法运行 CUDA benchmark。")
    except Exception as exc:
        return {
            "ok": False,
            "average_speedup": 0.0,
            "cases": [_empty_case_result(c.d, str(exc)) for c in cases],
            "error": str(exc),
        }

    with tempfile.TemporaryDirectory(prefix="mlsys_lora_bench_") as tmp:
        tmp_path = Path(tmp)
        cu_path = tmp_path / "optimized_lora.cu"
        cu_path.write_text(code, encoding="utf-8")
        build_dir = tmp_path / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            module = build_module(cu_path, build_dir)
        except Exception as exc:
            err = "编译失败: " + "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return {
                "ok": False,
                "average_speedup": 0.0,
                "cases": [_empty_case_result(c.d, err) for c in cases],
                "error": err,
            }

        results: list[dict[str, Any]] = []
        for case in cases:
            case_result = _empty_case_result(case.d)
            try:
                W, X, A, B = load_inputs(case.base_dir)
                with torch.no_grad():
                    y_student = module.forward(W, X, A, B)
                    y_ref = reference_impl(W, X, A, B)
                passed, max_abs_err, rel_l2_err = check_correctness(y_student, y_ref)
                case_result.update(
                    {
                        "correct": passed,
                        "max_abs_err": max_abs_err,
                        "rel_l2_err": rel_l2_err,
                    }
                )
                if passed:
                    student_ms = benchmark_fn(module.forward, W, X, A, B, warmup=warmup, iters=iters)
                    torch_ms = benchmark_fn(reference_impl, W, X, A, B, warmup=warmup, iters=iters)
                    speedup = torch_ms / student_ms if student_ms > 0 else 0.0
                    case_result.update(
                        {
                            "student_median_ms": student_ms,
                            "torch_median_ms": torch_ms,
                            "speedup": float(speedup),
                            "error": None,
                        }
                    )
                else:
                    case_result["error"] = "correctness check failed"
            except Exception as exc:
                case_result["error"] = "运行失败: " + "".join(traceback.format_exception_only(type(exc), exc)).strip()
            finally:
                results.append(case_result)

    avg = sum(float(r.get("speedup") or 0.0) for r in results) / len(results)
    ok = all(bool(r.get("correct")) and not r.get("error") for r in results)
    return {"ok": ok, "average_speedup": float(avg), "cases": results, "error": None if ok else "存在失败或未通过的 benchmark"}
