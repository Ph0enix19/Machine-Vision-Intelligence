from __future__ import annotations

import threading
import time
from typing import Any, Callable

import av
import requests

from dashboard.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from dashboard.utils import call_adapter


def make_video_frame_callback(
    adapter: Any,
    options: dict[str, Any],
) -> Callable[[av.VideoFrame], av.VideoFrame]:
    lock = threading.Lock()
    state: dict[str, Any] = {
        "last_processed_at": 0.0,
        "last_frame": None,
    }
    interval = 1.0 if "roboflow" in str(getattr(adapter, "method_name", "")).lower() else 0.15

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        cached = state["last_frame"]
        if cached is not None and now - float(state["last_processed_at"]) < interval:
            return av.VideoFrame.from_ndarray(cached, format="bgr24")
        if not lock.acquire(blocking=False):
            return av.VideoFrame.from_ndarray(cached if cached is not None else image_bgr, format="bgr24")

        try:
            result = call_adapter(adapter, image_bgr, frame=True, source_name="browser_webcam", **options)
            annotated = result.get("annotated_frame")
            if annotated is None:
                annotated = image_bgr
            state["last_frame"] = annotated
            state["last_processed_at"] = time.monotonic()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        finally:
            lock.release()

    return video_frame_callback


def get_rtc_configuration() -> dict[str, Any]:
    fallback = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return fallback

    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Tokens.json",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10,
        )
        if response.ok:
            ice_servers = response.json().get("ice_servers")
            if ice_servers:
                return {"iceServers": ice_servers}
    except (requests.RequestException, TypeError, ValueError):
        pass
    return fallback
