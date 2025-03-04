import numpy as np
import pandas as pd
from dataclasses import dataclass

from models.service_level_objectives import ServiceLevelObjectives


@dataclass
class SLOFulfillment:
    avg_actual_fps: float
    network_usage: float
    thermal_state: float
    stream_fulfillment: float
    avg_render_scale_factor: float
    qos: float
    qoe: float
    slo_fulfillment: float


    @staticmethod
    def calc(row: pd.Series, slos: ServiceLevelObjectives):
        def clip(value):
            return min(1., max(0., value))
    
        avg_actual_fps = clip(row["avg_actual_fps"] / slos.min_avg_fps)
        network_usage = clip(slos.max_network_usage / row["network_usage"])
        thermal_state = clip(slos.max_thermal_state / row["thermal_state"]) if row["thermal_state"] != 0 else 1.
        stream_fulfillment = clip(row["p_n_streams"] / slos.min_streams)
        avg_render_scale_factor = clip(slos.max_avg_render_scale_factor / row["avg_render_scale_factor"])

        qos_vars = [avg_actual_fps, network_usage, thermal_state]
        qoe_vars = [stream_fulfillment, avg_render_scale_factor]

        qos = sum(qos_vars) / len(qos_vars)
        qoe = sum(qoe_vars) / len(qoe_vars)

        slo_fulfillment = (qos + qoe) / 2

        return SLOFulfillment(
            avg_actual_fps,
            network_usage,
            thermal_state,
            stream_fulfillment,
            avg_render_scale_factor,
            qos,
            qoe,
            slo_fulfillment,
        )
    

    @staticmethod
    def calc_stats(sfs: list):
        mean_sf = SLOFulfillment(
            avg_actual_fps=np.mean([sf.avg_actual_fps for sf in sfs]),
            network_usage=np.mean([sf.network_usage for sf in sfs]),
            thermal_state=np.mean([sf.thermal_state for sf in sfs]),
            stream_fulfillment=np.mean([sf.stream_fulfillment for sf in sfs]),
            avg_render_scale_factor=np.mean([sf.avg_render_scale_factor for sf in sfs]),
            qoe=np.mean([sf.qoe for sf in sfs]),
            qos=np.mean([sf.qos for sf in sfs]),
            slo_fulfillment=np.mean([sf.slo_fulfillment for sf in sfs]),
        )
        std_sf = SLOFulfillment(
            avg_actual_fps=np.std([sf.avg_actual_fps for sf in sfs]),
            network_usage=np.std([sf.network_usage for sf in sfs]),
            thermal_state=np.std([sf.thermal_state for sf in sfs]),
            stream_fulfillment=np.std([sf.stream_fulfillment for sf in sfs]),
            avg_render_scale_factor=np.std([sf.avg_render_scale_factor for sf in sfs]),
            qoe=np.std([sf.qoe for sf in sfs]),
            qos=np.std([sf.qos for sf in sfs]),
            slo_fulfillment=np.std([sf.slo_fulfillment for sf in sfs]),
        )
        return mean_sf, std_sf
