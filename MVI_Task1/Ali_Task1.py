# ============================================================
# SEED CLASSIFIER — RULE-BASED COMPUTER VISION SYSTEM
# ============================================================

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


# ============================================================
# THRESHOLD VALUES USED FOR CLASSIFICATION
# ============================================================

# Midpoint values used in weighted scoring
AREA_MID = 63644.0
AR_MID = 1.5675
COMP_MID = 0.6762

# Hard limits for automatic classification
AREA_HARD_MATURE = 70000
AREA_HARD_IMMATURE = 55000

# Minimum score required to classify as mature
SCORE_THRESHOLD = 3

# Ignore very small contours (noise)
MIN_AREA = 800

# Output folder for saved videos
OUTPUT_DIR = "output_videos"

# Colors used for labels
COLOR = {
    "MATURE": (0, 200, 0),      # Green
    "IMMATURE": (0, 0, 220)    # Red
}


# ============================================================
# OPEN FILE BROWSER TO SELECT VIDEO
# ============================================================

def pick_video():

    # Create hidden tkinter window
    root = tk.Tk()
    root.withdraw()

    # Keep dialog on top
    root.attributes("-topmost", True)

    # Open file selection dialog
    path = filedialog.askopenfilename(
        title="Select seed video",
        filetypes=[
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return path


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(frame):

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE improves contrast and reduces lighting sensitivity
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Gaussian Blur removes noise and smooths image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles uneven lighting
    th1 = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        2
    )

    # OTSU automatically selects best threshold value
    _, th2 = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Combine both threshold results
    mask = cv2.bitwise_and(th1, th2)

    # Morphological operations clean small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Close small gaps and smooth object shape
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


# ============================================================
# EXTRACT SHAPE FEATURES FROM EACH SEED
# ============================================================

def extract_features(cnt):

    # Calculate contour area
    area = cv2.contourArea(cnt)

    # Ignore very small objects
    if area < MIN_AREA:
        return None

    # Calculate contour perimeter
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return None

    # Create bounding rectangle around seed
    x, y, w, h = cv2.boundingRect(cnt)

    # Determine seed dimensions
    length = float(max(w, h))
    width = float(min(w, h))

    # Shape measurements
    aspect_ratio = length / width if width != 0 else 0.0

    circularity = min(
        (4 * np.pi * area) / (perimeter ** 2),
        1.0
    )

    compactness = (
        area / (length * width)
        if (length * width) != 0 else 0.0
    )

    # Basic shape classification
    if circularity >= 0.85 and aspect_ratio <= 1.3:
        shape = "Round"

    elif circularity >= 0.60 and aspect_ratio <= 2.0:
        shape = "Oval"

    elif aspect_ratio > 2.0:
        shape = "Elongated"

    else:
        shape = "Irregular"

    # Return all extracted features
    return {
        "cnt": cnt,
        "bbox": (x, y, w, h),
        "area": round(float(area), 1),
        "length": round(length, 1),
        "width": round(width, 1),
        "perimeter": round(float(perimeter), 1),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 4),
        "compactness": round(compactness, 4),
        "shape": shape,
    }


# ============================================================
# WEIGHTED RULE-BASED CLASSIFIER
# ============================================================

def classify(f):

    area = f["area"]
    ar = f["aspect_ratio"]
    comp = f["compactness"]

    # Hard override rules
    if area >= AREA_HARD_MATURE:
        return "MATURE"

    if area <= AREA_HARD_IMMATURE:
        return "IMMATURE"

    # Weighted scoring system
    score = 0

    # Larger seeds gain points
    score += 2 if area >= AREA_MID else 0

    # Better aspect ratio gains points
    score += 2 if ar <= AR_MID else 0

    # Better compactness gains point
    score += 1 if comp <= COMP_MID else 0

    # Final decision
    label = (
        "MATURE"
        if score >= SCORE_THRESHOLD
        else "IMMATURE"
    )

    return label, f"score {score}/5"


# ============================================================
# DRAW RESULTS ON FRAME
# ============================================================

def draw_annotation(frame, f, label, detail):

    # Select label color
    c = COLOR[label]

    # Get contour
    cnt = f["cnt"]

    # Draw exact contour shape
    cv2.drawContours(frame, [cnt], -1, c, 2)

    x, y, w, h = f["bbox"]

    # Information displayed beside seed
    lines = [
        f"{label} [{detail}]",
        f"L:{f['length']} W:{f['width']}",
        f"AR:{f['aspect_ratio']}",
        f"Area:{f['area']}"
    ]

    # Draw text lines
    for i, line in enumerate(lines):

        ty = max(y - 10 - (3 - i) * 16, 12)

        cv2.putText(
            frame,
            line,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            c,
            1
        )


# ============================================================
# MAIN VIDEO PROCESSING
# ============================================================

def process_video(video_path):

    # Open selected video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError("Cannot open video")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output file path
    base = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{base}_classified.mp4"
    )

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # Process video frame-by-frame
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame
        mask = preprocess(frame)

        # Detect contours (seed outlines)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each detected seed
        for cnt in contours:

            # Extract features
            f = extract_features(cnt)

            if f is None:
                continue

            # Classify seed
            label, detail = classify(f)

            # Draw results
            draw_annotation(frame, f, label, detail)

        # Save processed frame
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()

    print("Processing complete.")
    print("Output saved:", output_path)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("SEED CLASSIFIER STARTED")

    # Open file browser
    video_path = pick_video()

    if not video_path:
        print("No file selected.")

    else:
        process_video(video_path)