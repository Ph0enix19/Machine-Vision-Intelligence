from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from dashboard.config import find_mvi_task1_files


def find_member_task1_file(member: str) -> Path | None:
    member_key = member.lower()
    matches = [
        path
        for path in find_mvi_task1_files()
        if member_key in path.name.lower() or member_key in str(path.parent).lower()
    ]
    return sorted(matches)[0] if matches else None


def import_task1_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
