from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "data.yaml"
PROJECT_DIR = ROOT / "runs" / "segment"
RUN_NAME = "bean_seg_v1"

EPOCHS = 300
IMG_SIZE = 640
AUTO_BATCH = -1
FALLBACK_BATCH = 4
MODEL_CANDIDATES = ("yolo11n-seg.pt", "yolov8n-seg.pt")


def choose_device() -> int | str:
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return 0
    except Exception as exc:
        print(f"Could not check CUDA with torch: {exc}")

    print("CUDA is not available; training will use CPU.")
    return "cpu"


def load_model():
    from ultralytics import YOLO

    last_error: Exception | None = None
    for weights in MODEL_CANDIDATES:
        try:
            print(f"Loading segmentation base model: {weights}")
            return YOLO(weights), weights
        except Exception as exc:
            last_error = exc
            print(f"WARNING: Could not load {weights}: {exc}")

    raise RuntimeError(f"Could not load any segmentation base model. Last error: {last_error}")


def main() -> int:
    if not DATA_YAML.exists():
        print(f"ERROR: Missing dataset YAML: {DATA_YAML}")
        print("Run scripts/check_dataset.py and confirm the dataset folder exists.")
        return 1

    model, base_weights = load_model()
    device = choose_device()

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
    print(f"Base weights: {base_weights}")
    print(f"Data: {DATA_YAML}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Run name: {RUN_NAME}")

    try:
        model.train(**train_args)
    except Exception as exc:
        print(f"WARNING: Training with auto batch failed: {exc}")
        print(f"Retrying with batch={FALLBACK_BATCH}")
        train_args["batch"] = FALLBACK_BATCH
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
