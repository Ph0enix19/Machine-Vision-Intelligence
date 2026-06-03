from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = [
    "White Kidney Bean",
    "Speckled Kidney Bean",
    "Dark Kidney Bean",
]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")
    return data


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return []


def resolve_path(value: Any, dataset_root: Path, yaml_dir: Path) -> Path | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None

    raw = Path(value)
    if raw.is_absolute():
        return raw

    base = dataset_root if dataset_root else yaml_dir
    return (base / raw).resolve()


def split_label_dir(image_dir: Path) -> Path:
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels"


def print_status(level: str, message: str) -> None:
    print(f"{level}: {message}")


def inspect_split(split_name: str, image_dir: Path | None, names: list[str]) -> tuple[int, int, int]:
    if image_dir is None:
        print_status("WARNING", f"{split_name}: path is not configured in data.yaml")
        return 0, 0, 1

    label_dir = split_label_dir(image_dir)
    if not image_dir.exists():
        print_status("WARNING", f"{split_name}: image directory does not exist: {image_dir}")
        return 0, 0, 1

    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    warnings = 0
    class_counts: Counter[int] = Counter()
    missing_labels: list[Path] = []
    empty_labels: list[Path] = []
    bad_class_ids: list[str] = []
    detection_like: list[str] = []
    malformed: list[str] = []
    out_of_range_coords: list[str] = []

    for image in images:
        label_path = label_dir / f"{image.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image.name)
            continue

        raw_text = label_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            empty_labels.append(label_path.name)
            continue

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                if len(parts) == 5:
                    detection_like.append(f"{label_path.name}:{line_number}")
                else:
                    malformed.append(f"{label_path.name}:{line_number}")
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                malformed.append(f"{label_path.name}:{line_number}")
                continue

            if class_id < 0 or class_id >= len(names):
                bad_class_ids.append(f"{label_path.name}:{line_number} -> {class_id}")
            else:
                class_counts[class_id] += 1

            try:
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                malformed.append(f"{label_path.name}:{line_number}")
                continue

            if any(value < 0.0 or value > 1.0 for value in coords):
                out_of_range_coords.append(f"{label_path.name}:{line_number}")

    print()
    print(f"[{split_name}]")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Image path: {image_dir}")
    print(f"Label path: {label_dir}")

    if images and len(labels) == len(images):
        print_status("PASS", f"{split_name}: image and label counts match")
    else:
        warnings += 1
        print_status("WARNING", f"{split_name}: image/label count mismatch")

    checks = [
        ("missing labels", missing_labels),
        ("empty labels", empty_labels),
        ("class IDs outside valid range", bad_class_ids),
        ("labels that look like detection boxes", detection_like),
        ("malformed segmentation rows", malformed),
        ("coordinates outside 0..1", out_of_range_coords),
    ]

    for label, items in checks:
        if items:
            warnings += 1
            sample = ", ".join(str(item) for item in items[:5])
            suffix = " ..." if len(items) > 5 else ""
            print_status("WARNING", f"{split_name}: {len(items)} {label}: {sample}{suffix}")
        else:
            print_status("PASS", f"{split_name}: no {label}")

    if class_counts:
        print("Instances per class:")
        for class_id, class_name in enumerate(names):
            print(f"  {class_id}: {class_name}: {class_counts[class_id]}")
    else:
        print_status("WARNING", f"{split_name}: no labeled instances found")
        warnings += 1

    return len(images), len(labels), warnings


def main() -> int:
    print("Kidney bean YOLO segmentation dataset check")
    print(f"Project root: {ROOT}")
    print(f"Data YAML: {DATA_YAML}")

    try:
        data = load_yaml(DATA_YAML)
    except Exception as exc:
        print_status("ERROR", str(exc))
        return 1

    names = normalize_names(data.get("names"))
    print()
    print("Classes:")
    for class_id, class_name in enumerate(names):
        print(f"  {class_id}: {class_name}")

    warnings = 0
    if data.get("task") == "segment":
        print_status("PASS", "task: segment is present")
    else:
        warnings += 1
        print_status("WARNING", "task: segment is missing or different")

    if names == EXPECTED_NAMES:
        print_status("PASS", "class names match the expected kidney bean order")
    else:
        warnings += 1
        print_status("WARNING", f"class names differ from expected order: {EXPECTED_NAMES}")

    yaml_dir = DATA_YAML.parent
    configured_root = data.get("path")
    if configured_root:
        dataset_root = Path(str(configured_root))
        if not dataset_root.is_absolute():
            dataset_root = (yaml_dir / dataset_root).resolve()
    else:
        dataset_root = yaml_dir

    split_paths = {
        "train": resolve_path(data.get("train"), dataset_root, yaml_dir),
        "valid": resolve_path(data.get("val") or data.get("valid"), dataset_root, yaml_dir),
        "test": resolve_path(data.get("test"), dataset_root, yaml_dir),
    }

    print()
    print("Configured paths:")
    for split_name, path in split_paths.items():
        print(f"  {split_name}: {path}")

    total_images = 0
    total_labels = 0
    for split_name, image_dir in split_paths.items():
        images, labels, split_warnings = inspect_split(split_name, image_dir, names)
        total_images += images
        total_labels += labels
        warnings += split_warnings

    print()
    print("Dataset summary")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    if warnings:
        print_status("WARNING", f"dataset check completed with {warnings} warning group(s)")
    else:
        print_status("PASS", "dataset is ready for YOLO segmentation training")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
