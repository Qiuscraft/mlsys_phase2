# mlsys-phase2

## 技术文档

https://bytedance.larkoffice.com/docx/RnvHdo1TMoM4pDxt867cDaELn6b

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

可用参数：

```text
--output-root PATH   输出根目录，默认 inputs
--seed INT           随机种子，默认 20260508
--scale FLOAT        随机张量缩放系数，默认 0.01
--overwrite          覆盖已存在文件
```

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

环境变量说明：

```text
OPENAI_API_KEY        OpenAI 兼容 API Key
OPENAI_BASE_URL       OpenAI 兼容 API Base URL，可选
OPENAI_MODEL          使用的模型，默认 gpt-4o
OPENAI_TEMPERATURE    采样温度，默认 0.2
MAX_OPT_ITERS         优化迭代次数；仅在未传 --max-opt-iters 时生效，默认 10
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

可用参数：

```text
--inputs-root PATH           输入根目录，默认项目根目录 inputs/
--output PATH                最佳代码输出路径，默认项目根目录 optimized.cu
--max-init-attempts INT      初始代码生成/修复最大尝试次数，默认 8
--max-opt-iters INT          优化迭代次数；不传时读取 MAX_OPT_ITERS，默认 10
--warmup INT                 benchmark warmup 次数，默认 10
--iters INT                  benchmark 正式计时次数，默认 50
--profile-iters INT          ncu profile 内部运行次数，默认 8
--time-limit-seconds FLOAT   Agent 总运行限时，默认 1800 秒；<=0 表示不限时
```

Agent 默认最多运行 30 分钟。超过 `--time-limit-seconds` 后会直接退出进程，退出码为 `124`。

ncu profile 每次会从输入目录中随机抽取 1 个 d 进行采集，并在日志摘要中输出 `sampled_d`。

通过正确性检查且平均 speedup 更高的代码会写入 `optimized.cu`。
