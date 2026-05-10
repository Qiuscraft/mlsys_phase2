#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 安装依赖
pip install -r requirements.txt

# 2. 确保目录存在
mkdir -p /workspace
mkdir -p logs

# 3. 后台启动文件监控
python file_watcher.py &
WATCHER_PID=$!

# 4. 前台运行 agent
set +e
python main.py "$@"
AGENT_EXIT_CODE=$?
set -e

# 5. 停止监控并做最终同步
kill "$WATCHER_PID" 2>/dev/null || true
wait "$WATCHER_PID" 2>/dev/null || true

exit $AGENT_EXIT_CODE
