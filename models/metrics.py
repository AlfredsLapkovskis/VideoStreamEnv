from dataclasses import dataclass


@dataclass
class Metrics:
    setting_id: int
    cpu_usage: float
    memory_usage: float
    network_usage: float
    avg_actual_fps: float
    avg_render_scale_factor: float
    thermal_state: int
    