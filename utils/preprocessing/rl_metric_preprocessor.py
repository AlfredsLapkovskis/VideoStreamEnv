import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.preprocessing.metric_preprocessor import MetricPreprocessor
from models.slo_fulfillment import SLOFulfillment
from utils.variables import *


class RLMetricPreprocessor(MetricPreprocessor):
    _CATEGORICAL_COLUMNS = {
        "setting_id",
        "thermal_state",
        VAR_P_N_STREAMS,
        VAR_P_RESOLUTION,
        VAR_P_FPS,
    }

    def __init__(self, session, training_df: pd.DataFrame):
        super().__init__(session)

        self._scaler = StandardScaler().fit(self._without_categorical(training_df))


    def preprocess(self, df):
        scaled_df = self._without_categorical(df)
        scaled_columns = scaled_df.columns
        scaled_df = self._scaler.transform(self._without_categorical(df))
        scaled_df = pd.DataFrame(scaled_df, columns=scaled_columns)

        return pd.DataFrame({
            VAR_P_FPS: list(df[VAR_P_FPS]),
            VAR_P_RESOLUTION: list(df[VAR_P_RESOLUTION]),
            VAR_P_N_STREAMS: list(df[VAR_P_N_STREAMS]),
            VAR_M_CPU_USAGE: list(scaled_df["cpu_usage"]),
            VAR_M_MEMORY_USAGE: list(scaled_df["memory_usage"]),
            VAR_S_AVG_ACTUAL_FPS: list(scaled_df["avg_actual_fps"]),
            VAR_S_NETWORK_USAGE: list(scaled_df["network_usage"]),
            VAR_S_AVG_RENDER_SCALE_FACTOR: list(scaled_df["avg_render_scale_factor"]),
            VAR_S_THERMAL_STATE: list(df["thermal_state"]),
            "reward": [
                SLOFulfillment.calc(row, self.session.slos).slo_fulfillment 
                for _, row in df.iterrows()
            ],
        })

    
    def _without_categorical(self, df):
        df = pd.DataFrame(df)
        for col in df.columns:
            if col in self._CATEGORICAL_COLUMNS:
                del df[col]
        return df
