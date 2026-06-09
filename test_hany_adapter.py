from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from dashboard.adapters.hany_adapter import HanyAdapter


def main() -> int:
    adapter = HanyAdapter()
    assert adapter.is_available(), adapter.availability_message()
    assert "Cloud secret detected" in adapter.availability_message()

    response = Mock()
    response.ok = True
    response.json.return_value = {
        "predictions": [
            {
                "x": 120,
                "y": 100,
                "width": 100,
                "height": 70,
                "class": "mature_healthy",
                "confidence": 0.91,
            }
        ]
    }
    image = np.full((240, 320, 3), 220, dtype=np.uint8)
    with patch("dashboard.adapters.hany_adapter.requests.post", return_value=response) as post:
        result = adapter.process_image(image, confidence=0.35)

    assert post.call_count == 1
    request = post.call_args
    assert request.args[0].endswith(adapter.model_id)
    assert request.kwargs["params"]["api_key"] == adapter.api_key
    assert result["metadata"]["credential_status"] == "Configured"
    assert result["metadata"]["roboflow_model"] == adapter.model_id
    assert result["summary"]["total_analyzed"] == 1
    assert result["summary"]["mature_count"] == 1
    assert result["summary"]["healthy_count"] == 1
    assert result["detections"][0]["roboflow_class"] == "mature_healthy"

    print("Hany secret detected: True")
    print("Hany model:", adapter.model_id)
    print("Mocked hosted inference detections:", result["summary"]["total_analyzed"])
    print("Mapped maturity:", result["detections"][0]["maturity_label"])
    print("Mapped health:", result["detections"][0]["health_label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
