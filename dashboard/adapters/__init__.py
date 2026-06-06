from __future__ import annotations

from dashboard.adapters.adonai_adapter import AdonaiAdapter
from dashboard.adapters.adonai_task1_adapter import AdonaiTask1Adapter
from dashboard.adapters.ali_adapter import AliAdapter
from dashboard.adapters.ali_task1_adapter import AliTask1Adapter
from dashboard.adapters.hany_adapter import HanyAdapter
from dashboard.adapters.hany_task1_adapter import HanyTask1Adapter
from dashboard.adapters.hemdan_task1_external_adapter import HemdanTask1ExternalAdapter
from dashboard.adapters.hemdan_yolo_adapter import HemdanYoloAdapter
from dashboard.adapters.tim_adapter import TimAdapter
from dashboard.adapters.tim_task1_adapter import TimTask1Adapter


def get_adapters():
    return [
        HemdanTask1ExternalAdapter(),
        AdonaiTask1Adapter(),
        AliTask1Adapter(),
        HanyTask1Adapter(),
        TimTask1Adapter(),
        HemdanYoloAdapter(),
        AdonaiAdapter(),
        AliAdapter(),
        HanyAdapter(),
        TimAdapter(),
    ]
