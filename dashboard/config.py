from __future__ import annotations

import os
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "dataset" / "data.yaml"
HEMDAN_YOLO_WEIGHTS = ROOT / "runs" / "segment" / "bean_seg_v1" / "weights" / "best.pt"
TIM_YOLO_WEIGHTS = ROOT / "group_members" / "tim" / "original" / "exp-2.pt"
TIM_TASK1_SOURCE = ROOT / "group_members" / "tim" / "task1_original" / "seed_inspection.py"
ALI_YOLO_WEIGHTS = ROOT / "group_members" / "ali" / "original" / "best.pt"

ADONAI_DEFAULT_WEIGHTS = ROOT / "group_members" / "adonai" / "original" / "weights" / "best.pt"
ADONAI_YOLO_WEIGHTS = Path(os.getenv("MVI_ADONAI_MODEL", str(ADONAI_DEFAULT_WEIGHTS)))
def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value)
    except (FileNotFoundError, KeyError, RuntimeError):
        pass
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with secrets_path.open("rb") as secrets_file:
                secret_value = tomllib.load(secrets_file).get(name)
            if secret_value is not None:
                return str(secret_value)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return default


HANY_ROBOFLOW_API_KEY = _setting("MVI_HANY_ROBOFLOW_API_KEY")
HANY_ROBOFLOW_MODEL_ID = _setting("MVI_HANY_ROBOFLOW_MODEL_ID", "mvi-task-2-dqpn6/2")
TWILIO_ACCOUNT_SID = _setting("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _setting("TWILIO_AUTH_TOKEN")
HIK_MVS_SDK_PATH = Path(
    _setting(
        "HIK_MVS_SDK_PATH",
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    )
)

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
