from __future__ import annotations

from perflowai.core.device import DeviceConfig


def bandwidth_Bps(device_config: DeviceConfig) -> float:
    """Convert DeviceConfig.memory_bandwidth to B/s.

    DeviceConfig.memory_bandwidth is documented as GB/s; convert to B/s.
    If a caller already passes B/s (very large number), keep it as-is.
    """

    bw = float(device_config.memory_bandwidth)
    if bw < 1e6:
        return bw * 1e9
    return bw
