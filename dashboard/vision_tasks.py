from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 35, 25]), np.array([95, 255, 255]))
    green_ratio = float(np.count_nonzero(green_mask)) / float(green_mask.size)

    if green_ratio > 0.10:
        mask = cv2.bitwise_not(green_mask)
        kernel_size = 5
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]
        smooth_background = cv2.GaussianBlur(gray, (0, 0), 21)
        local_dark = cv2.subtract(smooth_background, gray)
        contrast_mask = cv2.inRange(local_dark, 12, 255)
        colourful = cv2.inRange(saturation, 45, 255)
        mask = cv2.bitwise_or(contrast_mask, colourful)
        kernel_size = 3

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.medianBlur(mask, 5)


def contours_from_mask(image_bgr: np.ndarray, mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return contours

    _, sure_foreground = cv2.threshold(distance, 0.32 * distance.max(), 255, cv2.THRESH_BINARY)
    sure_foreground = np.uint8(sure_foreground)
    labels_count, markers = cv2.connectedComponents(sure_foreground)
    if labels_count <= 2:
        return contours

    markers = markers + 1
    unknown = cv2.subtract(mask, sure_foreground)
    markers[unknown == 255] = 0
    cv2.watershed(image_bgr.copy(), markers)

    split_contours: list[np.ndarray] = []
    for label in range(2, labels_count + 1):
        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        object_mask[markers == label] = 255
        object_contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not object_contours:
            continue
        contour = max(object_contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_area:
            split_contours.append(contour)
    return split_contours if len(split_contours) > len(contours) else contours


def contour_measurements(contour: np.ndarray) -> dict[str, float | int | str]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    length = float(max(w, h))
    width = float(min(w, h))
    aspect_ratio = length / width if width else None
    circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else None
    compactness = circularity
    equivalent_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else None
    shape = classify_shape(aspect_ratio, circularity)
    return {
        "area": area,
        "perimeter": perimeter,
        "box_x": int(x),
        "box_y": int(y),
        "box_width": int(w),
        "box_height": int(h),
        "length": length,
        "width": width,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "compactness": compactness,
        "equivalent_diameter": equivalent_diameter,
        "shape": shape,
    }


def classify_shape(aspect_ratio: float | None, circularity: float | None) -> str:
    if aspect_ratio is None:
        return "Unknown"
    if circularity is not None and circularity < 0.45:
        return "Irregular"
    if aspect_ratio < 1.25:
        return "Round"
    if aspect_ratio < 2.1:
        return "Oval"
    return "Elongated"


def bean_class_from_colour(image_bgr: np.ndarray, contour: np.ndarray) -> str:
    mask = contour_mask(image_bgr, contour)
    mean_bgr = cv2.mean(image_bgr, mask=mask)[:3]
    mean_hsv = cv2.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV), mask=mask)[:3]
    mean_gray = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
    hue, saturation, value = mean_hsv
    if mean_gray > 110 and saturation < 130:
        return "White Kidney Bean"
    if mean_gray < 95 or value < 95:
        return "Dark Kidney Bean"
    if 0 <= hue <= 30 or saturation >= 55:
        return "Speckled Kidney Bean"
    return "White Kidney Bean"


def colour_health_record(image_bgr: np.ndarray, contour: np.ndarray) -> dict[str, Any]:
    mask = contour_mask(image_bgr, contour)
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean_r, mean_g, mean_b = cv2.mean(rgb_image, mask=mask)[:3]
    mean_h, mean_s, mean_v = cv2.mean(hsv_image, mask=mask)[:3]
    pixels = rgb_image[mask == 255]
    color_uniformity = float(np.mean(np.std(pixels, axis=0))) if pixels.size else None

    dark_pixels = cv2.bitwise_and(cv2.inRange(gray, 0, 65), mask)
    dark_patch_ratio = float(np.count_nonzero(dark_pixels)) / max(float(np.count_nonzero(mask)), 1.0)
    discoloration_status = "Dark Patch" if dark_patch_ratio > 0.08 else "None"
    health_label = "Discoloration" if discoloration_status != "None" else "Healthy"
    maturity_label = maturity_from_hsv(mean_h, mean_s, mean_v)

    return {
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "mean_h": float(mean_h),
        "mean_s": float(mean_s),
        "mean_v": float(mean_v),
        "color_uniformity": color_uniformity,
        "discoloration_status": discoloration_status,
        "dark_patch_ratio": dark_patch_ratio,
        "maturity_label": maturity_label,
        "health_label": health_label,
    }


def maturity_from_hsv(mean_h: float, mean_s: float, mean_v: float) -> str:
    if mean_v < 70:
        return "Overripe"
    if mean_s < 45 and mean_v > 150:
        return "Mature"
    if mean_s < 100:
        return "Semi-Mature"
    return "Immature"


def quality_record(image_bgr: np.ndarray, contour: np.ndarray, confidence: float | None = None) -> dict[str, Any]:
    measurements = contour_measurements(contour)
    mask = contour_mask(image_bgr, contour)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = float(measurements["area"]) / hull_area
    dark_pixels = cv2.bitwise_and(cv2.inRange(gray, 0, 65), mask)
    dark_ratio = float(np.count_nonzero(dark_pixels)) / max(float(np.count_nonzero(mask)), 1.0)

    defect_type = "None"
    if solidity < 0.55:
        defect_type = "Damaged"
    if measurements["circularity"] is not None and float(measurements["circularity"]) < 0.42:
        defect_type = "Broken"
    if dark_ratio > 0.14:
        defect_type = "Moldy"

    quality_status = "Healthy" if defect_type == "None" else "Defective"
    return {
        "quality_status": quality_status,
        "defect_type": defect_type,
        "crack": defect_type == "Crack",
        "broken": defect_type == "Broken",
        "moldy": defect_type == "Moldy",
        "damaged": defect_type == "Damaged",
        "area": measurements["area"],
        "confidence": confidence,
    }


def texture_record(image_bgr: np.ndarray, contour: np.ndarray, confidence: float | None = None) -> dict[str, Any]:
    mask = contour_mask(image_bgr, contour)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = float(np.count_nonzero(cv2.bitwise_and(edges, mask))) / max(float(np.count_nonzero(mask)), 1.0)

    values = gray[mask == 255]
    entropy = grayscale_entropy(values)
    energy = grayscale_energy(values)
    texture_score = min(1.0, edge_ratio * 5.0 + (entropy / 8.0) * 0.35)
    if texture_score < 0.32:
        texture_label = "Smooth"
        irregularity = "Low"
        pattern = "Even"
    elif texture_score < 0.62:
        texture_label = "Medium"
        irregularity = "Medium"
        pattern = "Normal"
    else:
        texture_label = "Rough"
        irregularity = "High"
        pattern = "Uneven"

    return {
        "texture_label": texture_label,
        "texture_type": "Surface pattern",
        "texture_score": texture_score,
        "irregularity_level": irregularity,
        "surface_pattern": pattern,
        "entropy": entropy,
        "energy": energy,
        "confidence": confidence,
    }


def grayscale_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def grayscale_energy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    return float(np.sum(probs * probs))


def contour_mask(image_bgr: np.ndarray, contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def contour_from_box(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)


def draw_label(image_bgr: np.ndarray, text: str, x: int, y: int, colour: tuple[int, int, int]) -> None:
    y = max(y, 22)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image_bgr, (x, y - text_h - baseline - 7), (x + text_w + 8, y), colour, -1)
    cv2.putText(image_bgr, text, (x + 4, y - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def counts_from_records(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(record.get(key) for record in records if record.get(key)))

