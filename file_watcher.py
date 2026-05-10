#!/usr/bin/env python3
"""文件监控脚本：将 agent 产出实时映射到 /workspace/ 目标路径。"""

import signal
import shutil
import sys
import time
from pathlib import Path

# ============ 配置 ============
PROJECT_ROOT = Path(__file__).resolve().parent
POLL_INTERVAL = 2  # 秒

# 源文件
SRC_OPTIMIZED_CU = PROJECT_ROOT / "optimized.cu"
SRC_LOG_DIR = PROJECT_ROOT / "logs"

# 目标文件
DST_OPTIMIZED_CU = Path("/workspace/optimized_lora.cu")
DST_LOG = Path("/workspace/output.log")

# ============ 状态 ============
_shutdown = False
_last_cu_mtime = 0.0
_last_log_file = None
_last_log_size = 0


def handle_signal(signum, frame):
    global _shutdown
    _shutdown = True


def find_latest_log():
    """在 logs/ 目录下找到最新的 .txt 文件。"""
    if not SRC_LOG_DIR.is_dir():
        return None
    log_files = sorted(SRC_LOG_DIR.glob("*.txt"))
    return log_files[-1] if log_files else None


def sync_optimized_cu():
    """同步 optimized.cu -> /workspace/optimized_lora.cu"""
    global _last_cu_mtime

    if not SRC_OPTIMIZED_CU.exists():
        return

    try:
        current_mtime = SRC_OPTIMIZED_CU.stat().st_mtime
    except OSError:
        return

    if current_mtime != _last_cu_mtime:
        try:
            shutil.copy2(SRC_OPTIMIZED_CU, DST_OPTIMIZED_CU)
            _last_cu_mtime = current_mtime
        except OSError as e:
            print(f"[watcher] 同步 optimized.cu 失败: {e}", file=sys.stderr)


def sync_log_file():
    """增量同步日志文件 -> /workspace/output.log"""
    global _last_log_file, _last_log_size

    latest = find_latest_log()
    if latest is None:
        return

    # 日志文件切换时重置状态
    if latest != _last_log_file:
        _last_log_file = latest
        _last_log_size = 0
        try:
            DST_LOG.write_bytes(b"")
        except OSError:
            pass

    try:
        current_size = latest.stat().st_size
    except OSError:
        return

    if current_size <= _last_log_size:
        return

    try:
        with open(latest, "rb") as f:
            f.seek(_last_log_size)
            new_data = f.read()

        with open(DST_LOG, "ab") as f:
            f.write(new_data)

        _last_log_size = current_size
    except OSError as e:
        print(f"[watcher] 同步日志失败: {e}", file=sys.stderr)


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    DST_OPTIMIZED_CU.parent.mkdir(parents=True, exist_ok=True)
    DST_LOG.parent.mkdir(parents=True, exist_ok=True)

    print(f"[watcher] 启动文件监控 (轮询间隔={POLL_INTERVAL}s)", file=sys.stderr)

    while not _shutdown:
        sync_optimized_cu()
        sync_log_file()

        for _ in range(int(POLL_INTERVAL / 0.2)):
            if _shutdown:
                break
            time.sleep(0.2)

    # 最终同步
    time.sleep(0.5)
    sync_optimized_cu()
    sync_log_file()
    print("[watcher] 监控结束。", file=sys.stderr)


if __name__ == "__main__":
    main()
