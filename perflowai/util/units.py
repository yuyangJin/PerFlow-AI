from __future__ import annotations

from perflowai.core.device import DeviceConfig

__all__ = ['bandwidth_Bps', 'intra_node_bandwidth_Bps', 'inter_node_bandwidth_Bps']

def _gbps_to_Bps(bw_gbps: float) -> float:
    """Convert a bandwidth value to B/s.

    Convention in this repo:
    - Most bandwidth fields on DeviceConfig are documented as GB/s.
    - For convenience we allow callers to pass already-in-B/s values.
    """

    bw = float(bw_gbps)
    return bw * 1e9


def bandwidth_Bps(device_config: DeviceConfig) -> float:
    """Convert DeviceConfig.memory_bandwidth to B/s.

    DeviceConfig.memory_bandwidth is documented as GB/s; convert to B/s.
    If a caller already passes B/s (very large number), keep it as-is.
    """

    return _gbps_to_Bps(device_config.memory_bandwidth)


def intra_node_bandwidth_Bps(device_config: DeviceConfig) -> float:
    """Convert DeviceConfig.intra_node_bandwidth to B/s."""

    return _gbps_to_Bps(device_config.intra_node_bandwidth)


def inter_node_bandwidth_Bps(device_config: DeviceConfig) -> float:
    """Convert DeviceConfig.inter_node_bandwidth to B/s."""

    return _gbps_to_Bps(device_config.inter_node_bandwidth)
