import sys
import cv2
import numpy as np
import tempfile
import os
from ctypes import *

from inference_sdk import InferenceHTTPClient

# ============================================================
# IMPORT MVS SDK
# ============================================================

sys.path.append(
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
)

from MvCameraControl_class import *

# ============================================================
# ROBOFLOW CONFIG
# ============================================================

API_KEY = os.getenv("MVI_HANY_ROBOFLOW_API_KEY", "")
MODEL_ID = os.getenv("MVI_HANY_ROBOFLOW_MODEL_ID", "mvi-task-2-dqpn6/2")

CONFIDENCE_THRESHOLD = 0.4

# ============================================================
# COLOURS
# ============================================================

COLORS = [
    (255, 82, 82),
    (82, 255, 121),
    (82, 182, 255),
    (255, 210, 82),
    (210, 82, 255),
]

def get_color(class_name):
    return COLORS[hash(class_name) % len(COLORS)]

# ============================================================
# ROBOFLOW CLIENT
# ============================================================

if not API_KEY:
    raise RuntimeError("Set MVI_HANY_ROBOFLOW_API_KEY before running this script.")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

# ============================================================
# SEARCH FOR CAMERA
# ============================================================

deviceList = MV_CC_DEVICE_INFO_LIST()

tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

print("=" * 60)
print("SEARCHING FOR MVS CAMERA")
print("=" * 60)

ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

if ret != 0:
    print(f"Camera enumeration failed! Error: 0x{ret:X}")
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("No camera found!")
    sys.exit()

print(f"Found {deviceList.nDeviceNum} camera(s)")

# ============================================================
# CREATE CAMERA HANDLE
# ============================================================

stDeviceList = cast(
    deviceList.pDeviceInfo[0],
    POINTER(MV_CC_DEVICE_INFO)
).contents

cam = MvCamera()

ret = cam.MV_CC_CreateHandle(stDeviceList)

if ret != 0:
    print(f"Create handle failed! Error: 0x{ret:X}")
    sys.exit()

# ============================================================
# OPEN CAMERA
# ============================================================

ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

if ret != 0:
    print(f"Open camera failed! Error: 0x{ret:X}")
    sys.exit()

print("✓ Camera opened successfully")

# ============================================================
# GIGE OPTIMIZATION
# ============================================================

nPacketSize = cam.MV_CC_GetOptimalPacketSize()

if int(nPacketSize) > 0:

    ret = cam.MV_CC_SetIntValue(
        "GevSCPSPacketSize",
        nPacketSize
    )

    if ret == 0:
        print(f"✓ Packet size set: {nPacketSize}")
    else:
        print(f"⚠ Packet size failed: 0x{ret:X}")

else:
    print("⚠ Could not get optimal packet size")

# ============================================================
# TRIGGER MODE OFF
# ============================================================

ret = cam.MV_CC_SetEnumValue("TriggerMode", 0)

if ret == 0:
    print("✓ Trigger mode OFF")
else:
    print(f"⚠ Trigger mode failed: 0x{ret:X}")

# ============================================================
# PIXEL FORMAT
# ============================================================

formats_to_try = [
    ("BayerRG8", PixelType_Gvsp_BayerRG8),
    ("BayerGB8", PixelType_Gvsp_BayerGB8),
    ("Mono8", PixelType_Gvsp_Mono8),
    ("RGB8", PixelType_Gvsp_RGB8_Packed),
]

current_format = None

for fmt_name, fmt_value in formats_to_try:

    ret = cam.MV_CC_SetEnumValue(
        "PixelFormat",
        fmt_value
    )

    print(f"Trying {fmt_name} -> ret = 0x{ret:X}")

    if ret == 0:
        current_format = fmt_name
        print(f"✓ Using pixel format: {fmt_name}")
        break

if current_format is None:
    print("✗ No pixel format could be set")
    sys.exit()

# ============================================================
# EXPOSURE SETTINGS
# ============================================================

try:
    ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)

    if ret == 0:
        print("✓ Auto exposure OFF")

except:
    pass

try:
    ret = cam.MV_CC_SetFloatValue(
        "ExposureTime",
        5000.0
    )

    if ret == 0:
        print("✓ Exposure set")

except:
    pass

# ============================================================
# GET PAYLOAD SIZE
# ============================================================

stParam = MVCC_INTVALUE()

memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

ret = cam.MV_CC_GetIntValue(
    "PayloadSize",
    stParam
)

if ret != 0:
    print("Get payload size failed!")
    sys.exit()

nPayloadSize = stParam.nCurValue

print(f"Payload size: {nPayloadSize}")

# ============================================================
# BUFFER
# ============================================================

data_buf = (c_ubyte * nPayloadSize)()

# ============================================================
# START GRABBING
# ============================================================

ret = cam.MV_CC_StartGrabbing()

if ret != 0:
    print(f"Start grabbing failed! Error: 0x{ret:X}")
    sys.exit()

print("\n" + "=" * 60)
print("MVS CAMERA READY")
print(f"Pixel Format: {current_format}")
print("Press Q to quit")
print("=" * 60)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    ret = cam.MV_CC_GetOneFrameTimeout(
        byref(data_buf),
        nPayloadSize,
        stFrameInfo,
        2000
    )

    if ret != 0:
        print(f"Failed to grab frame: 0x{ret:X}")
        continue

    width = stFrameInfo.nWidth
    height = stFrameInfo.nHeight

    # ========================================================
    # FRAME CONVERSION
    # ========================================================

    try:

        frame_data = np.frombuffer(
            data_buf,
            dtype=np.uint8
        )

        # RGB
        if current_format == "RGB8":

            frame = frame_data[
                :width * height * 3
            ].reshape((height, width, 3))

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

        # MONO
        elif current_format == "Mono8":

            frame = frame_data[
                :width * height
            ].reshape((height, width))

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR
            )

        # BAYER
        else:

            frame = frame_data[
                :width * height
            ].reshape((height, width))

            if current_format == "BayerRG8":

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BAYER_RG2BGR
                )

            else:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BAYER_GB2BGR
                )

    except Exception as e:
        print(f"Frame conversion failed: {e}")
        continue

    # ========================================================
    # ROBOFLOW INFERENCE
    # ========================================================

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        cv2.imwrite(tmp_path, frame)

        result = client.infer(
            tmp_path,
            model_id=MODEL_ID
        )

    except Exception as e:

        cv2.putText(
            frame,
            f"Inference Error: {e}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "MVS Roboflow Detection",
            frame
        )

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    finally:

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    predictions = result.get(
        "predictions",
        []
    )

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    for pred in predictions:

        confidence = pred.get(
            "confidence",
            0
        )

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        cx = int(pred["x"])
        cy = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2

        class_name = pred.get(
            "class",
            "object"
        )

        color = get_color(class_name)

        label = f"{class_name} {confidence:.0%}"

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Text background
        (lw, lh), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - lh - baseline - 6),
            (x1 + lw + 4, y1),
            color,
            cv2.FILLED
        )

        # Label
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

    # ========================================================
    # HUD
    # ========================================================

    hud = (
        f"Detections: {len(predictions)} | "
        f"Format: {current_format} | "
        f"Q to Quit"
    )

    cv2.putText(
        frame,
        hud,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MVS Roboflow Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
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
