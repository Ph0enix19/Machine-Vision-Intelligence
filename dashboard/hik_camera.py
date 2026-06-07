from __future__ import annotations

import importlib
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from dashboard.config import HIK_MVS_SDK_PATH


class HikCameraError(RuntimeError):
    pass


class HikMVSCamera:
    def __init__(
        self,
        device_index: int = 0,
        exposure_time: float = 5000.0,
        sdk_path: Path = HIK_MVS_SDK_PATH,
    ) -> None:
        self.device_index = int(device_index)
        self.exposure_time = float(exposure_time)
        self.sdk_path = sdk_path
        self._sdk: ModuleType | None = None
        self._camera = None
        self._payload_size = 0
        self._data_buffer = None
        self._opened = False

    def open(self) -> None:
        sdk = load_hik_sdk(self.sdk_path)
        device_list = sdk.MV_CC_DEVICE_INFO_LIST()
        transport_types = sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE
        result = sdk.MvCamera.MV_CC_EnumDevices(transport_types, device_list)
        if result != 0:
            raise HikCameraError(f"HIK device enumeration failed with code 0x{result:X}.")
        if device_list.nDeviceNum == 0:
            raise HikCameraError("No HIK MVS cameras were detected.")
        if self.device_index < 0 or self.device_index >= device_list.nDeviceNum:
            raise HikCameraError(
                f"HIK camera index {self.device_index} is invalid; "
                f"{device_list.nDeviceNum} camera(s) detected."
            )

        device_info = cast(
            device_list.pDeviceInfo[self.device_index],
            POINTER(sdk.MV_CC_DEVICE_INFO),
        ).contents
        camera = sdk.MvCamera()
        result = camera.MV_CC_CreateHandle(device_info)
        if result != 0:
            raise HikCameraError(f"HIK create-handle failed with code 0x{result:X}.")

        self._sdk = sdk
        self._camera = camera
        try:
            result = camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0)
            if result != 0:
                raise HikCameraError(f"HIK open-device failed with code 0x{result:X}.")

            result = camera.MV_CC_SetEnumValue("PixelFormat", sdk.PixelType_Gvsp_RGB8_Packed)
            if result != 0:
                raise HikCameraError(f"HIK RGB8 pixel-format setup failed with code 0x{result:X}.")

            result = camera.MV_CC_SetEnumValue("ExposureAuto", 0)
            if result != 0:
                raise HikCameraError(f"HIK auto-exposure disable failed with code 0x{result:X}.")

            result = camera.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if result != 0:
                raise HikCameraError(f"HIK exposure setup failed with code 0x{result:X}.")

            payload = sdk.MVCC_INTVALUE()
            memset(byref(payload), 0, sizeof(sdk.MVCC_INTVALUE))
            result = camera.MV_CC_GetIntValue("PayloadSize", payload)
            if result != 0:
                raise HikCameraError(f"HIK payload-size query failed with code 0x{result:X}.")

            self._payload_size = int(payload.nCurValue)
            self._data_buffer = (c_ubyte * self._payload_size)()
            result = camera.MV_CC_StartGrabbing()
            if result != 0:
                raise HikCameraError(f"HIK start-grabbing failed with code 0x{result:X}.")
            self._opened = True
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms: int = 1000) -> np.ndarray:
        if not self._opened or self._sdk is None or self._camera is None or self._data_buffer is None:
            raise HikCameraError("HIK camera is not open.")

        frame_info = self._sdk.MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(self._sdk.MV_FRAME_OUT_INFO_EX))
        result = self._camera.MV_CC_GetOneFrameTimeout(
            self._data_buffer,
            self._payload_size,
            frame_info,
            int(timeout_ms),
        )
        if result != 0:
            raise HikCameraError(f"HIK frame capture failed with code 0x{result:X}.")

        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        expected_bytes = width * height * 3
        if width <= 0 or height <= 0 or expected_bytes > self._payload_size:
            raise HikCameraError("HIK camera returned invalid RGB8 frame dimensions.")

        frame_rgb = np.frombuffer(
            self._data_buffer,
            dtype=np.uint8,
            count=expected_bytes,
        ).reshape((height, width, 3))
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        camera = self._camera
        if camera is not None:
            if self._opened:
                camera.MV_CC_StopGrabbing()
            camera.MV_CC_CloseDevice()
            camera.MV_CC_DestroyHandle()
        self._opened = False
        self._camera = None
        self._data_buffer = None
        self._payload_size = 0


def load_hik_sdk(sdk_path: Path = HIK_MVS_SDK_PATH) -> ModuleType:
    module_file = sdk_path / "MvCameraControl_class.py"
    if not module_file.exists():
        raise HikCameraError(
            "HIK MVS Python SDK was not found. Set HIK_MVS_SDK_PATH to the "
            "MvImport folder containing MvCameraControl_class.py."
        )
    path_text = str(sdk_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    try:
        return importlib.import_module("MvCameraControl_class")
    except Exception as exc:
        raise HikCameraError(f"HIK MVS Python SDK import failed: {exc}") from exc


def hik_camera_status(sdk_path: Path = HIK_MVS_SDK_PATH) -> tuple[bool, str, int]:
    try:
        sdk = load_hik_sdk(sdk_path)
        device_list = sdk.MV_CC_DEVICE_INFO_LIST()
        result = sdk.MvCamera.MV_CC_EnumDevices(
            sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE,
            device_list,
        )
        if result != 0:
            return False, f"Device enumeration failed with code 0x{result:X}.", 0
        count = int(device_list.nDeviceNum)
        if count == 0:
            return False, "HIK MVS SDK loaded, but no camera was detected.", 0
        return True, f"HIK MVS SDK loaded; {count} camera(s) detected.", count
    except HikCameraError as exc:
        return False, str(exc), 0
