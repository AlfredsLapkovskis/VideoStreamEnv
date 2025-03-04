from dataclasses import dataclass


@dataclass
class ServiceLevelObjectives:
    max_network_usage: int
    min_avg_fps: float
    min_streams: int
    max_avg_render_scale_factor: float
    max_thermal_state: int