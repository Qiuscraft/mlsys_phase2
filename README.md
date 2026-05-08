# mlsys-phase2

LoRA CUDA 自动优化 Agent。目标算子：

```text
Y = W X + A (B^T X)
```

其中 `W, X` 为 `d x d`，`A, B` 为 `d x 16`，`d ∈ {3584, 3840, 4096, 4352, 4608}`，全部为 CUDA contiguous float32。

## 输入目录

默认读取：

```text
inputs/d3584/{W.pt,X.pt,A.pt,B.pt}
inputs/d3840/{W.pt,X.pt,A.pt,B.pt}
inputs/d4096/{W.pt,X.pt,A.pt,B.pt}
inputs/d4352/{W.pt,X.pt,A.pt,B.pt}
inputs/d4608/{W.pt,X.pt,A.pt,B.pt}
```

生成方法：

```shell
python generate_lora_inputs.py --output-root inputs --overwrite
```

## MCP 工具

启动 stdio MCP server：

```bash
mlsys-mcp
```

提供工具：

- `benchmark_lora(code, inputs_root=None, warmup=10, iters=50)`
- `profile_lora_ncu(code, inputs_root=None, iters=8)`

## 自动优化 Agent

配置 OpenAI 兼容 API：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # 可选
export OPENAI_MODEL=...      # 默认 gpt-4o
```

运行：

```bash
uv run python -m mlsys_phase2.agent --inputs-root inputs --max-opt-iters 10
```

通过正确性检查且平均 speedup 更高的代码会写入 `optimized.cu`。
