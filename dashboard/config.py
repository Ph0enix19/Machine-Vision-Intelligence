from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "dataset" / "data.yaml"
HEMDAN_YOLO_WEIGHTS = ROOT / "runs" / "segment" / "bean_seg_v1" / "weights" / "best.pt"
TIM_YOLO_WEIGHTS = ROOT / "group_members" / "tim" / "original" / "exp-2.pt"
ALI_YOLO_WEIGHTS = ROOT / "group_members" / "ali" / "original" / "best.pt"

ADONAI_DEFAULT_WEIGHTS = ROOT / "group_members" / "adonai" / "original" / "weights" / "best.pt"
ADONAI_YOLO_WEIGHTS = Path(os.getenv("MVI_ADONAI_MODEL", str(ADONAI_DEFAULT_WEIGHTS)))
HANY_ROBOFLOW_API_KEY = os.getenv("MVI_HANY_ROBOFLOW_API_KEY", "")
HANY_ROBOFLOW_MODEL_ID = os.getenv("MVI_HANY_ROBOFLOW_MODEL_ID", "mvi-task-2-dqpn6/2")

MVI_TASK1_CANDIDATE_DIRS = [
    ROOT / "MVI_Task1",
    ROOT.parent / "MVI_Task1",
]

OUTPUTS_DIR = ROOT / "outputs"
OUTPUT_VIDEOS_DIR = OUTPUTS_DIR / "videos"
OUTPUT_RESULTS_DIR = OUTPUTS_DIR / "results"

BEAN_CLASSES = [
    "White Kidney Bean",
    "Speckled Kidney Bean",
    "Dark Kidney Bean",
]

SORTING_CLASS_ROWS = [
    "White Kidney Bean",
    "Speckled Kidney Bean",
    "Dark Kidney Bean",
]

DEFECT_TYPES = [
    "None",
    "Crack",
    "Broken",
    "Moldy",
    "Damaged",
    "Discoloration",
    "Irregular",
    "N/A",
]

QUALITY_TYPES = ["Healthy", "Defective", "N/A"]
MATURITY_TYPES = ["Mature", "Immature", "N/A"]

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMG_SIZE = 640
DEFAULT_DEVICE = ""
VIDEO_PREVIEW_EVERY_N_FRAMES = 10


def ensure_output_dirs() -> None:
    OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_mvi_task1_files() -> list[Path]:
    files: list[Path] = []
    for folder in MVI_TASK1_CANDIDATE_DIRS:
        if folder.exists():
            files.extend(
                path for path in folder.rglob("*.py") if "__pycache__" not in path.parts and path.is_file()
            )
    return sorted(files)
