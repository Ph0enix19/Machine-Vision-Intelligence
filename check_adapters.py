from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dashboard.adapters import get_adapters
from dashboard.config import ALI_YOLO_WEIGHTS


def adapter_weights(adapter: object) -> str:
    path = getattr(adapter, "weights_path", None)
    if path is not None:
        return str(path)
    if getattr(adapter, "member", "") == "Ali" and "YOLO" in getattr(adapter, "method_name", ""):
        return str(ALI_YOLO_WEIGHTS)
    return "N/A"


def main() -> int:
    print(f"Dashboard root: {ROOT}")
    print(f"Configured ALI_YOLO_WEIGHTS: {ALI_YOLO_WEIGHTS}")
    print(f"Ali weights exist: {ALI_YOLO_WEIGHTS.exists()}")
    print()

    failed = False
    for adapter in get_adapters():
        try:
            available = adapter.is_available()
            print(f"name: {adapter.name}")
            print(f"available: {available}")
            print(f"message: {adapter.availability_message()}")
            print(f"weights: {adapter_weights(adapter)}")
            print()
            failed = failed or not available
        except Exception as exc:
            failed = True
            print(f"name: {getattr(adapter, 'name', type(adapter).__name__)}")
            print("available: ERROR")
            print(f"message: {type(exc).__name__}: {exc}")
            print(f"weights: {adapter_weights(adapter)}")
            print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
