from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .utils import project_root

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
LOG_DIR_NAME = "logs"
CONSOLE_COLORS = {
    "INFO": "\033[34m",
    "WARN": "\033[33m",
    "ERROR": "\033[32m",
}
RESET_COLOR = "\033[0m"


class BeijingFormatter(logging.Formatter):
    """Formatter that always renders timestamps in Beijing time."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        return f"{dt:%Y-%m-%d %H:%M:%S}.{int(record.msecs):03d}"


class ColorFormatter(BeijingFormatter):
    """Console formatter with per-level ANSI colors."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        color = CONSOLE_COLORS.get(record.levelname)
        if color is None:
            return rendered
        return f"{color}{rendered}{RESET_COLOR}"


class MlsysStreamHandler(logging.StreamHandler):
    _mlsys_phase2_handler = True


class MlsysFileHandler(logging.FileHandler):
    _mlsys_phase2_handler = True


LOG_FORMAT = "%(levelname)s %(asctime)s %(pathname)s:%(lineno)d %(message)s"


def _log_file_path() -> Path:
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")
    return project_root() / LOG_DIR_NAME / f"{timestamp}.txt"


def setup_logging(log_to_file: bool = True) -> Path | None:
    """Configure project logging and optionally create the Agent log file."""
    logging.addLevelName(logging.WARNING, "WARN")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_mlsys_phase2_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    console_handler = MlsysStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    log_path: Path | None = None
    if log_to_file:
        log_path = _log_file_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = MlsysFileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(BeijingFormatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

    return log_path
