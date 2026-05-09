from __future__ import annotations

from typing import Any

from . import D_VALUES, RANK
from .utils import dumps_result

MAX_PROMPT_ERROR_CHARS = 4000


def _drop_none_items(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _compact_error(error: Any) -> str | None:
    if not error:
        return None
    text = str(error)
    if len(text) <= MAX_PROMPT_ERROR_CHARS:
        return text
    return text[:MAX_PROMPT_ERROR_CHARS] + "...[truncated]"


def _metric_has_signal(metric_summary: dict[str, Any]) -> bool:
    values = metric_summary.get("values")
    if isinstance(values, list):
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        if numeric_values:
            return any(value != 0 for value in numeric_values)

    candidates = [metric_summary.get("min"), metric_summary.get("max"), metric_summary.get("avg")]
    numeric_candidates = [value for value in candidates if isinstance(value, (int, float))]
    return bool(numeric_candidates) and any(value != 0 for value in numeric_candidates)


def _compact_metrics_for_prompt(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    if not isinstance(metrics, dict):
        return compact
    for name, summary in metrics.items():
        if not isinstance(summary, dict) or not _metric_has_signal(summary):
            continue
        compact[name] = _drop_none_items(
            {
                "count": summary.get("count"),
                "min": summary.get("min"),
                "max": summary.get("max"),
                "avg": summary.get("avg"),
            }
        )
    return compact


def summarize_benchmark_for_prompt(result: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    unique_errors: list[str] = []

    for case in result.get("cases") or []:
        if not isinstance(case, dict):
            continue
        compact_error = _compact_error(case.get("error"))
        if compact_error and compact_error not in unique_errors:
            unique_errors.append(compact_error)
        cases.append(
            _drop_none_items(
                {
                    "d": case.get("d"),
                    "correct": case.get("correct"),
                    "speedup": case.get("speedup"),
                    "student_median_ms": case.get("student_median_ms"),
                    "torch_median_ms": case.get("torch_median_ms"),
                    "max_abs_err": case.get("max_abs_err"),
                    "rel_l2_err": case.get("rel_l2_err"),
                    "error": compact_error,
                }
            )
        )

    return _drop_none_items(
        {
            "ok": result.get("ok"),
            "average_speedup": result.get("average_speedup"),
            "error": _compact_error(result.get("error")),
            "cases": cases,
            "unique_errors": unique_errors or None,
        }
    )


def summarize_profile_for_prompt(result: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    for case in result.get("cases") or []:
        if not isinstance(case, dict):
            continue
        summary = case.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        kernels = [
            _drop_none_items({"name": kernel.get("name"), "count": kernel.get("count")})
            for kernel in summary.get("kernels") or []
            if isinstance(kernel, dict)
        ]
        cases.append(
            _drop_none_items(
                {
                    "d": case.get("d"),
                    "ok": case.get("ok"),
                    "returncode": case.get("returncode"),
                    "error": _compact_error(case.get("error")),
                    "kernel_count": summary.get("kernel_count"),
                    "kernels": kernels,
                    "metrics": _compact_metrics_for_prompt(summary.get("metrics") or {}),
                }
            )
        )

    return _drop_none_items(
        {
            "ok": result.get("ok"),
            "sampled_d": result.get("sampled_d"),
            "error": _compact_error(result.get("error")),
            "cases": cases,
        }
    )


def benchmark_prompt_json(result: dict[str, Any]) -> str:
    return dumps_result(summarize_benchmark_for_prompt(result))


def profile_prompt_json(result: dict[str, Any]) -> str:
    return dumps_result(summarize_profile_for_prompt(result))

SYSTEM_PROMPT = f"""
你是 CUDA 性能优化专家。你只能输出完整的 optimized_lora.cu 源码，不要输出 Markdown、解释或省略号。
目标算子：Y = W X + A (B^T X)。W,X 为 dxd，A,B 为 dxr，r={RANK}，d 属于 {list(D_VALUES)}，所有 tensor 是 CUDA contiguous float32。
必须实现并通过 PYBIND11_MODULE 暴露：
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B);
源文件必须可单独编译，只使用标准 CUDA/C++/PyTorch extension 头文件。
正确性标准：torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)。
优先优化所有 d 的平均 speedup；不要依赖固定单一 d；发生不确定时选择安全正确的实现。
""".strip()

INITIAL_USER_PROMPT = f"""
请生成第一版 optimized_lora.cu CUDA 实现，只输出完整 .cu 源码，不要输出 Markdown、解释或省略号。

任务目标：实现并优化 LoRA 算子：
Y = W X + A (B^T X)

输入张量约束：
- W: d x d，CUDA contiguous float32
- X: d x d，CUDA contiguous float32
- A: d x {RANK}，CUDA contiguous float32
- B: d x {RANK}，CUDA contiguous float32
- rank r = {RANK}
- benchmark 覆盖 d in {list(D_VALUES)}，实现不要只针对单一 d 写死逻辑，应兼顾所有 d 的平均性能。

必须暴露如下接口：
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B);

编译与依赖要求：
- 必须通过 PYBIND11_MODULE(...) 暴露 forward。
- 单文件可编译，不依赖项目内其他文件。
- 只使用标准 CUDA/C++ 头文件和系统已有的 PyTorch extension 头文件。

正确性要求：
- 参考实现为 torch 语义：W @ X + A @ (B.transpose(0, 1).contiguous() @ X)。
- 必须满足 torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)。

性能目标：
- 在保证正确性的前提下优化所有 d 的平均 speedup。
- 第一版可以优先选择稳妥正确、容易通过编译和正确性检查的实现，再做适度优化。
""".strip()


def repair_prompt(code: str, bench_result: str) -> str:
    return f"""
上一版代码没有通过编译、运行或正确性检查。请基于错误信息重新生成完整 .cu 代码，只输出代码。

上一版代码：
{code}

benchmark 摘要：
{bench_result}
""".strip()


def optimize_prompt(code: str, profile_result: str, best_speedup: float) -> str:
    return f"""
当前代码已正确，平均 speedup={best_speedup:.6f}。请根据 ncu profile 结果优化性能，目标是提升 5 个 d 的平均 speedup。
只能输出完整 .cu 代码。必须保持接口和正确性要求不变。

当前代码：
{code}

ncu profile 摘要：
{profile_result}
""".strip()
