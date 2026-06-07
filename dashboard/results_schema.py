from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

import numpy as np


TASKS = {
    "I": {
        "member": "Hemdan",
        "task_name": "Seed Classification",
        "badge": "Task I",
        "table_columns": [
            "id",
            "class_name",
            "confidence",
            "area",
            "box_x",
            "box_y",
            "box_width",
            "box_height",
            "mask_area",
        ],
    },
    "II": {
        "member": "Adonai",
        "task_name": "Quality Inspection",
        "badge": "Task II",
        "table_columns": [
            "id",
            "quality_status",
            "defect_type",
            "crack",
            "broken",
            "moldy",
            "damaged",
            "area",
            "confidence",
        ],
    },
    "III": {
        "member": "Ali",
        "task_name": "Seed Growth Inspection using Measurement",
        "badge": "Task III",
        "table_columns": [
            "id",
            "length",
            "width",
            "area",
            "perimeter",
            "aspect_ratio",
            "circularity",
            "compactness",
            "equivalent_diameter",
            "shape",
        ],
    },
    "IV": {
        "member": "Hany",
        "task_name": "Maturity and Health Condition",
        "badge": "Task IV",
        "table_columns": [
            "id",
            "mean_r",
            "mean_g",
            "mean_b",
            "mean_h",
            "mean_s",
            "mean_v",
            "color_uniformity",
            "discoloration_status",
            "dark_patch_ratio",
            "maturity_label",
            "health_label",
            "roboflow_class",
            "confidence",
        ],
    },
    "V": {
        "member": "Tim",
        "task_name": "Texture Inspection",
        "badge": "Task V",
        "table_columns": [
            "id",
            "texture_label",
            "texture_type",
            "texture_score",
            "irregularity_level",
            "surface_pattern",
            "entropy",
            "energy",
            "confidence",
        ],
    },
}


def make_task_result(
    *,
    annotated_frame: np.ndarray | None,
    member: str,
    task_id: str,
    task_name: str,
    method: str,
    summary: dict[str, Any] | None = None,
    task_outputs: dict[str, Any] | None = None,
    detections: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "annotated_frame": annotated_frame,
        "member": member,
        "task_id": task_id,
        "task_name": task_name,
        "method": method,
        "summary": summary or {},
        "task_outputs": task_outputs or {},
        "detections": detections or [],
        "metadata": metadata or {},
    }


def make_unavailable_result(
    *,
    annotated_frame: np.ndarray | None,
    member: str,
    task_id: str,
    task_name: str,
    method: str,
    reason: str,
) -> dict[str, Any]:
    return make_task_result(
        annotated_frame=annotated_frame,
        member=member,
        task_id=task_id,
        task_name=task_name,
        method=method,
        summary={"primary_result": "Unavailable"},
        task_outputs={"unavailable_reason": reason},
        detections=[],
        metadata={"error": reason, "status": "Unavailable"},
    )


def ensure_result(result: dict[str, Any] | None, default_frame: np.ndarray | None = None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return make_unavailable_result(
            annotated_frame=default_frame,
            member="Unknown",
            task_id="",
            task_name="Unknown Task",
            method="Unknown",
            reason="Adapter did not return a result dictionary.",
        )

    if "task_id" in result and "task_name" in result:
        result.setdefault("annotated_frame", default_frame)
        result.setdefault("member", TASKS.get(result.get("task_id", ""), {}).get("member", "Unknown"))
        result.setdefault("method", result.get("metadata", {}).get("method", "Unknown"))
        result.setdefault("summary", {})
        result.setdefault("task_outputs", {})
        result.setdefault("detections", [])
        result.setdefault("metadata", {})
        return result

    metadata = result.get("metadata") or {}
    member = metadata.get("member", "Unknown")
    method = metadata.get("method", "Legacy adapter")
    detections = result.get("detections") or []
    counts = result.get("counts") or {}
    total = sum(int(value or 0) for value in counts.values()) or len(detections)
    return make_task_result(
        annotated_frame=result.get("annotated_frame", default_frame),
        member=member,
        task_id="",
        task_name="Legacy Output",
        method=method,
        summary={"total_detected": total, "primary_result": f"{total} detection(s)"},
        task_outputs={"counts": counts, "defect_counts": result.get("defect_counts") or {}},
        detections=detections,
        metadata=metadata,
    )


def task_columns(task_id: str) -> list[str]:
    return TASKS.get(task_id, {}).get("table_columns", [])


def detections_to_rows(result: dict[str, Any] | None, source: str = "") -> list[dict[str, Any]]:
    result = ensure_result(result)
    columns = task_columns(result.get("task_id", ""))
    rows: list[dict[str, Any]] = []
    for detection in result.get("detections", []):
        if columns:
            row = {column: detection.get(column) for column in columns if column in detection}
        else:
            row = dict(detection)
        row["source"] = source or result.get("method", "")
        row["member"] = result.get("member", "")
        row["task"] = result.get("task_name", "")
        row["method"] = result.get("method", "")
        rows.append(row)
    return rows


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    keys = list(rows[0].keys())
    empty_keys = set()
    for key in keys:
        values = [row.get(key) for row in rows]
        if all(value is None or value == "" for value in values):
            empty_keys.add(key)
    return [{key: value for key, value in row.items() if key not in empty_keys} for row in rows]


def summarize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    result = ensure_result(result)
    return result.get("summary", {})


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    task_counter: Counter[str] = Counter()
    row_count = 0
    summaries: list[dict[str, Any]] = []
    for result in results:
        result = ensure_result(result)
        task_counter[result.get("task_name", "Unknown")] += 1
        row_count += len(result.get("detections", []))
        summaries.append(
            {
                "member": result.get("member", ""),
                "task": result.get("task_name", ""),
                "method": result.get("method", ""),
                **(result.get("summary") or {}),
            }
        )
    return {"tasks": dict(task_counter), "row_count": row_count, "summaries": summaries}


def build_video_result(
    *,
    template_result: dict[str, Any],
    rows: list[dict[str, Any]],
    annotated_frame: np.ndarray | None,
    frames_processed: int,
    fps: float,
    summary_mode: str = "latest_frame",
) -> dict[str, Any]:
    template_result = ensure_result(template_result)
    task_id = template_result.get("task_id", "")
    summary, task_outputs = summarize_rows_for_task(task_id, rows)
    summary["frames_processed"] = frames_processed
    summary["primary_result"] = "Latest frame summary; cumulative frame CSV saved separately."
    return make_task_result(
        annotated_frame=annotated_frame,
        member=template_result.get("member", ""),
        task_id=task_id,
        task_name=template_result.get("task_name", ""),
        method=f"{template_result.get('method', '')} - Video",
        summary=summary,
        task_outputs=task_outputs,
        detections=rows,
        metadata={
            **(template_result.get("metadata") or {}),
            "fps": fps,
            "frames_processed": frames_processed,
            "summary_mode": summary_mode,
        },
    )


def summarize_rows_for_task(task_id: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_id == "I":
        counts = Counter(row.get("class_name") for row in rows if row.get("class_name"))
        confidences = [float(row["confidence"]) for row in rows if _is_number(row.get("confidence"))]
        summary = {
            "total_detected": len(rows),
            "white_kidney_count": counts.get("White Kidney Bean", 0),
            "speckled_kidney_count": counts.get("Speckled Kidney Bean", 0),
            "dark_kidney_count": counts.get("Dark Kidney Bean", 0),
            "average_confidence": mean(confidences) if confidences else None,
            "primary_result": f"{len(counts)} bean class(es) detected",
        }
        return summary, {"class_counts": dict(counts)}

    if task_id == "II":
        quality = Counter(row.get("quality_status") for row in rows if row.get("quality_status"))
        defects = Counter(row.get("defect_type") for row in rows if row.get("defect_type"))
        summary = {
            "total_inspected": len(rows),
            "healthy_count": quality.get("Healthy", 0),
            "defective_count": quality.get("Defective", 0),
            "crack_count": defects.get("Crack", 0),
            "broken_count": defects.get("Broken", 0),
            "moldy_count": defects.get("Moldy", 0),
            "damaged_count": defects.get("Damaged", 0),
            "unknown_count": defects.get("Unknown", 0),
        }
        return summary, {"quality_counts": dict(quality), "defect_type_counts": dict(defects)}

    if task_id == "III":
        lengths = [float(row["length"]) for row in rows if _is_number(row.get("length"))]
        widths = [float(row["width"]) for row in rows if _is_number(row.get("width"))]
        areas = [float(row["area"]) for row in rows if _is_number(row.get("area"))]
        aspect_ratios = [float(row["aspect_ratio"]) for row in rows if _is_number(row.get("aspect_ratio"))]
        shapes = Counter(row.get("shape") for row in rows if row.get("shape"))
        summary = {
            "total_measured": len(rows),
            "average_length": mean(lengths) if lengths else None,
            "average_width": mean(widths) if widths else None,
            "average_area": mean(areas) if areas else None,
            "average_aspect_ratio": mean(aspect_ratios) if aspect_ratios else None,
            "most_common_shape": shapes.most_common(1)[0][0] if shapes else None,
        }
        return summary, {"measurement_units": "pixels", "shape_counts": dict(shapes)}

    if task_id == "IV":
        maturity = Counter(row.get("maturity_label") for row in rows if row.get("maturity_label"))
        health = Counter(row.get("health_label") for row in rows if row.get("health_label"))
        discoloration = sum(1 for row in rows if row.get("discoloration_status") not in {None, "", "None"})
        dark_patch = sum(1 for row in rows if _is_number(row.get("dark_patch_ratio")) and float(row["dark_patch_ratio"]) > 0.08)
        summary = {
            "total_analyzed": len(rows),
            "mature_count": maturity.get("Mature", 0),
            "semi_mature_count": maturity.get("Semi-Mature", 0),
            "immature_count": maturity.get("Immature", 0),
            "overripe_count": maturity.get("Overripe", 0),
            "healthy_count": health.get("Healthy", 0),
            "discoloration_count": discoloration,
            "dark_patch_count": dark_patch,
        }
        return summary, {"maturity_counts": dict(maturity), "health_counts": dict(health)}

    if task_id == "V":
        textures = Counter(row.get("texture_label") for row in rows if row.get("texture_label"))
        scores = [float(row["texture_score"]) for row in rows if _is_number(row.get("texture_score"))]
        summary = {
            "total_analyzed": len(rows),
            "smooth_count": textures.get("Smooth", 0),
            "medium_texture_count": textures.get("Medium", 0) + textures.get("Normal", 0),
            "rough_count": textures.get("Rough", 0),
            "irregular_texture_count": textures.get("Irregular", 0),
            "average_texture_score": mean(scores) if scores else None,
        }
        return summary, {"texture_counts": dict(textures)}

    return {"total_records": len(rows)}, {}


def _is_number(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
