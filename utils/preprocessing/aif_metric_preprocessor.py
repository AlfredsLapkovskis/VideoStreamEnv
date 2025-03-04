import pandas as pd
import numpy as np

from utils.preprocessing.metric_preprocessor import MetricPreprocessor
from models.service_level_objectives import ServiceLevelObjectives
from utils.variables import *
from models.slo_fulfillment import SLOFulfillment


class AIFMetricPreprocessor(MetricPreprocessor):
    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        updated_df = pd.DataFrame(df)
        slos: ServiceLevelObjectives = self.session.slos

        updated_df[VAR_M_CPU_USAGE] = pd.cut(updated_df["cpu_usage"], bins=[.0, .2, .4, .6, .8, float("inf")], labels=range(1, 6), include_lowest=True)
        updated_df[VAR_M_MEMORY_USAGE] = pd.cut(updated_df["memory_usage"], bins=[.0, .2, .4, .6, .8, float("inf")], labels=range(1, 6), include_lowest=True)

        zeros = [0] * len(updated_df)
        for field in SLO_LIST:
            updated_df[field] = zeros

        for i, row in df.iterrows():
            def discretize(value):
                thresholds = [.2, .4, .6, .8, 1.]
                for i, t in enumerate(thresholds):
                    if value < t:
                        return i
                return len(thresholds)

            sf = SLOFulfillment.calc(row, slos)

            updated_df.at[i, VAR_S_AVG_ACTUAL_FPS] = discretize(sf.avg_actual_fps)
            updated_df.at[i, VAR_S_NETWORK_USAGE] = discretize(sf.network_usage)
            updated_df.at[i, VAR_S_STREAM_FULFILLMENT] = discretize(sf.stream_fulfillment)
            updated_df.at[i, VAR_S_AVG_RENDER_SCALE_FACTOR] = discretize(sf.avg_render_scale_factor)
            updated_df.at[i, VAR_S_THERMAL_STATE] = discretize(sf.thermal_state)
        
        column_names = set(FULL_VAR_LIST)
        for column in list(updated_df.columns):
            if column not in column_names:
                del updated_df[column]

        return updated_df
