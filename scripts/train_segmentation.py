from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "data.yaml"
PROJECT_DIR = ROOT / "runs" / "segment"
RUN_NAME = "bean_seg_v1"

EPOCHS = 300
IMG_SIZE = 640
AUTO_BATCH = -1
BASE_MODEL = "yolo11n-seg.pt"


def require_cuda() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for this training configuration.")

    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    return 0


def main() -> int:
    if not DATA_YAML.exists():
        print(f"ERROR: Missing dataset YAML: {DATA_YAML}")
        print("Run scripts/check_dataset.py and confirm the dataset folder exists.")
        return 1

    from ultralytics import YOLO

    device = require_cuda()
    print(f"Loading segmentation base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    train_args = {
        "task": "segment",
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": AUTO_BATCH,
        "device": device,
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
    }

    print()
    print("Starting YOLO segmentation training")
    print(f"Base weights: {BASE_MODEL}")
    print(f"Data: {DATA_YAML}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Run name: {RUN_NAME}")

    model.train(**train_args)

    weights_dir = PROJECT_DIR / RUN_NAME / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    print()
    print("Training complete")
    print(f"best.pt: {best_pt}")
    print(f"last.pt: {last_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
