# -*- coding: utf-8 -*-
"""
LIVE SEED SEGMENTATION — YOLOv8 SEG + HIKVISION MV CAMERA
=========================================================
- Uses Hikvision MVS SDK directly
- Uses YOLOv8 segmentation model
- Draws OUTLINES around seeds instead of boxes
- Counts MATURE / IMMATURE seeds
"""

import sys
import os
import cv2
import numpy as np
from ctypes import *
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"runs\segment\runs\seed\train-5\weights\best.pt"   # CHANGE THIS
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
IMGSZ = 640

WINDOW_NAME = "Live Seed Segmentation"

# ============================================================
# LOAD MVS SDK
# ============================================================

sys.path.append(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")

try:
    from MvCameraControl_class import *
    print("✓ MVS SDK Loaded")
except Exception as e:
    print(f"✗ Failed loading MVS SDK: {e}")
    sys.exit()

# ============================================================
# LOAD YOLO SEGMENTATION MODEL
# ============================================================

print("\nLoading YOLOv8 segmentation model...")

try:
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded")
    print("Classes:", model.names)

except Exception as e:
    print(f"✗ Failed loading model: {e}")
    sys.exit()

# ============================================================
# ENUMERATE CAMERA
# ============================================================

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

if ret != 0:
    print("✗ Camera enumeration failed")
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("✗ No camera found")
    sys.exit()

print(f"✓ Found {deviceList.nDeviceNum} camera(s)")

# ============================================================
# CREATE CAMERA
# ============================================================

cam = MvCamera()

stDeviceList = cast(
    deviceList.pDeviceInfo[0],
    POINTER(MV_CC_DEVICE_INFO)
).contents

ret = cam.MV_CC_CreateHandle(stDeviceList)

if ret != 0:
    print("✗ Create handle failed")
    sys.exit()

ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

if ret != 0:
    print("✗ Open device failed")
    sys.exit()

print("✓ Camera opened")

cam.MV_CC_SetEnumValue("ExposureAuto", 0)
cam.MV_CC_SetFloatValue("ExposureTime", 3000.0)
cam.MV_CC_SetFloatValue("Gain", 0.0)

# ============================================================
# SET RGB FORMAT
# ============================================================

try:
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
    print("✓ RGB8 format enabled")
except:
    print("! Could not set RGB8")

# ============================================================
# GET PAYLOAD SIZE
# ============================================================

stParam = MVCC_INTVALUE()
memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)

if ret != 0:
    print("✗ Failed getting payload size")
    sys.exit()

nPayloadSize = stParam.nCurValue
data_buf = (c_ubyte * nPayloadSize)()

# ============================================================
# START GRABBING
# ============================================================

ret = cam.MV_CC_StartGrabbing()

if ret != 0:
    print("✗ Failed starting camera")
    sys.exit()

print("✓ Camera streaming started")

# ============================================================
# COLORS
# ============================================================

COLORS = {
    "MATURE": (0, 255, 0),
    "IMMATURE": (0, 0, 255),
}

# ============================================================
# MAIN LOOP
# ============================================================

print("\nPress Q to quit")

while True:

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    ret = cam.MV_CC_GetOneFrameTimeout(
        data_buf,
        nPayloadSize,
        stFrameInfo,
        1000
    )

    if ret == 0:

        width = stFrameInfo.nWidth
        height = stFrameInfo.nHeight

        frame = np.frombuffer(
            data_buf,
            dtype=np.uint8,
            count=width * height * 3
        )

        frame = frame.reshape((height, width, 3))

        # RGB -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ====================================================
        # YOLOv8 SEGMENTATION
        # ====================================================

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False
        )

        mature_count = 0
        immature_count = 0

        result = results[0]

        if result.masks is not None:

            masks = result.masks.xy
            classes = result.boxes.cls
            confs = result.boxes.conf

            for i, mask in enumerate(masks):

                points = np.array(mask, dtype=np.int32)

                cls = int(classes[i])

                conf = float(confs[i])

                label = model.names[cls].upper()

                # ============================================
                # CLASS COLORS
                # ============================================

                if label == "MATURE":

                    mature_count += 1
                    color = COLORS["MATURE"]

                else:

                    immature_count += 1
                    color = COLORS["IMMATURE"]

                # ============================================
                # TRANSPARENT FILL
                # ============================================

                overlay = frame.copy()

                cv2.fillPoly(
                    overlay,
                    [points],
                    color
                )

                alpha = 0.25

                frame = cv2.addWeighted(
                    overlay,
                    alpha,
                    frame,
                    1 - alpha,
                    0
                )

                # ============================================
                # OUTLINE
                # ============================================

                cv2.polylines(
                    frame,
                    [points],
                    isClosed=True,
                    color=color,
                    thickness=3
                )

                # ============================================
                # LABEL
                # ============================================

                x, y = points[0]

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # ====================================================
        # INFO PANEL
        # ====================================================

        total = mature_count + immature_count

        cv2.putText(
            frame,
            f"MATURE: {mature_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"IMMATURE: {immature_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"TOTAL: {total}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(WINDOW_NAME, frame)

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# ============================================================
# CLEANUP
# ============================================================

print("\nCleaning up...")

cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()

cv2.destroyAllWindows()

print("✓ Done")