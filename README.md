# mlsys-phase2

## 环境

```shell
uv venv
uv sync
source ./.venv/bin/activate
```

或

```shell
pip install -r requirements.txt
```

## 目标

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
python -m mlsys_phase2.mcp_server
```

提供工具：

- `benchmark_lora(code, inputs_root=None, warmup=10, iters=50)`
- `profile_lora_ncu(code, inputs_root=None, iters=8)`

## 自动优化 Agent

配置 OpenAI 兼容 API。Agent 启动时会自动读取项目根目录 `.env`，且已通过 shell/export 设置的环境变量优先于 `.env`：

```bash
# .env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...        # 可选
OPENAI_MODEL=gpt-4o        # 默认 gpt-4o
OPENAI_TEMPERATURE=0.2     # 默认 0.2
MAX_OPT_ITERS=10           # --max-opt-iters 未传入时生效
```

也可以继续直接通过 shell 配置：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # 可选
export OPENAI_MODEL=...      # 默认 gpt-4o
```

运行：

```bash
python -m mlsys_phase2.agent --inputs-root inputs --max-opt-iters 10
```

通过正确性检查且平均 speedup 更高的代码会写入 `optimized.cu`。
