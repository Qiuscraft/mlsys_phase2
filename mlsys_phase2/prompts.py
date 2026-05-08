from __future__ import annotations

from . import D_VALUES, RANK

SYSTEM_PROMPT = f"""
你是 CUDA 性能优化专家。你只能输出完整的 optimized_lora.cu 源码，不要输出 Markdown、解释或省略号。
目标算子：Y = W X + A (B^T X)。W,X 为 dxd，A,B 为 dxr，r={RANK}，d 属于 {list(D_VALUES)}，所有 tensor 是 CUDA contiguous float32。
必须实现并通过 PYBIND11_MODULE 暴露：
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B);
源文件必须可单独编译，只使用标准 CUDA/C++/PyTorch extension 头文件。
正确性标准：torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)。
优先优化所有 d 的平均 speedup；不要依赖固定单一 d；发生不确定时选择安全正确的实现。
""".strip()

INITIAL_USER_PROMPT = """
请生成第一版 CUDA 实现。建议先确保正确性，再做适度优化。只输出完整 .cu 代码。
""".strip()


def repair_prompt(code: str, bench_result: str) -> str:
    return f"""
上一版代码没有通过编译、运行或正确性检查。请基于错误信息重新生成完整 .cu 代码，只输出代码。

上一版代码：
{code}

benchmark 结果：
{bench_result}
""".strip()


def optimize_prompt(code: str, bench_result: str, profile_result: str, best_speedup: float) -> str:
    return f"""
当前代码已正确，平均 speedup={best_speedup:.6f}。请根据 benchmark 和 ncu profile 结果优化性能，目标是提升 5 个 d 的平均 speedup。
只能输出完整 .cu 代码。必须保持接口和正确性要求不变。

当前代码：
{code}

benchmark 结果：
{bench_result}

ncu profile 结果：
{profile_result}
""".strip()
