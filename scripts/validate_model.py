from __future__ import annotations

from pathlib import Path
from pprint import pprint

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "data.yaml"
MODEL_PATH = ROOT / "runs" / "segment" / "bean_seg_v1" / "weights" / "best.pt"
IMG_SIZE = 640


def metric_value(metrics, group: str, attr: str):
    group_obj = getattr(metrics, group, None)
    if group_obj is None:
        return None
    return getattr(group_obj, attr, None)


def print_metric(label: str, value) -> bool:
    if value is None:
        return False
    try:
        print(f"{label}: {float(value):.4f}")
    except (TypeError, ValueError):
        print(f"{label}: {value}")
    return True


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: Missing trained weights: {MODEL_PATH}")
        print("Run training first: python scripts/train_segmentation.py")
        return 1

    if not DATA_YAML.exists():
        print(f"ERROR: Missing dataset YAML: {DATA_YAML}")
        return 1

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    metrics = model.val(data=str(DATA_YAML), task="segment", imgsz=IMG_SIZE, split="val")

    print()
    print("Validation metrics")
    printed = False
    printed |= print_metric("Mask mAP50-95", metric_value(metrics, "seg", "map"))
    printed |= print_metric("Mask mAP50", metric_value(metrics, "seg", "map50"))
    printed |= print_metric("Mask precision", metric_value(metrics, "seg", "mp"))
    printed |= print_metric("Mask recall", metric_value(metrics, "seg", "mr"))
    printed |= print_metric("Box mAP50-95", metric_value(metrics, "box", "map"))
    printed |= print_metric("Box mAP50", metric_value(metrics, "box", "map50"))
    printed |= print_metric("Box precision", metric_value(metrics, "box", "mp"))
    printed |= print_metric("Box recall", metric_value(metrics, "box", "mr"))

    results_dict = getattr(metrics, "results_dict", None)
    if results_dict:
        print()
        print("Full results dictionary")
        pprint(results_dict)
        printed = True

    if not printed:
        print("Could not find standard metric attributes. Full metrics object:")
        pprint(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
