import pandas as pd
import math

from utils.preprocessing.metric_preprocessor import MetricPreprocessor


class HeatingMetricPreprocessorWrapper(MetricPreprocessor):

    _MAX_T = 1.0

    def __init__(self, metric_preprocessor: MetricPreprocessor, k=None):
        super().__init__(metric_preprocessor.session)

        self._metric_preprocessor = metric_preprocessor
        self._time = 0
        self._t_next = 0.
        self._k = k if k is not None else 0.07


    def preprocess(self, df: pd.DataFrame):
        df = pd.DataFrame(df)

        t_next = self._t_next

        for i, row in df.iterrows():
            t_curr = t_next
            df.at[i, "thermal_state"] = self.temperature_to_thermal_state(t_curr)
            t_env = self.network_to_temperature(row["network_usage"])
            t_next = t_env + (t_curr - t_env) * math.e ** (-self._k)

        self._t_next = t_next

        return self._metric_preprocessor.preprocess(df)
    

    @staticmethod
    def network_to_temperature(network):
        return min(HeatingMetricPreprocessorWrapper._MAX_T, 0.364 * math.e ** (0.05 * (network / 1024 / 1024)))
    

    @staticmethod
    def temperature_to_thermal_state(t):
        thresholds = [0.25, 0.5, 0.75]
        for i, threshold in enumerate(thresholds):
            if t < threshold:
                return i
        return len(thresholds)
