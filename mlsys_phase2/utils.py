from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import D_VALUES


@dataclass(frozen=True)
class CaseInput:
    d: int
    base_dir: Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_inputs_root(inputs_root: str | os.PathLike[str] | None = None) -> Path:
    if inputs_root is None:
        return project_root() / "inputs"
    return Path(inputs_root).expanduser().resolve()


def discover_cases(inputs_root: str | os.PathLike[str] | None = None) -> list[CaseInput]:
    root = resolve_inputs_root(inputs_root)
    return [CaseInput(d=d, base_dir=root / f"d{d}") for d in D_VALUES]


def validate_case_dirs(inputs_root: str | os.PathLike[str] | None = None) -> list[str]:
    missing: list[str] = []
    for case in discover_cases(inputs_root):
        for name in ("W.pt", "X.pt", "A.pt", "B.pt"):
            p = case.base_dir / name
            if not p.exists():
                missing.append(str(p))
    return missing


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(make_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True)


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        m = re.match(r"^```(?:cuda|cu|cpp|c\+\+)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip() + "\n"
    return text


def require_executable(name: str) -> str | None:
    return shutil.which(name)


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
