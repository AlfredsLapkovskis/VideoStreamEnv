import pandas as pd
from itertools import cycle

from utils.data_iterators.data_iterator import DataIterator
from utils.variables import *


class DataFrameDataIterator(DataIterator):

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int=1,
        default_configuration=None,
    ):
        self.df = df
        self._configuration_dfs = {}
        self._default_configuration = default_configuration
        if not default_configuration:
            self._default_configuration = {
                VAR_P_N_STREAMS: VALUES_P_N_STREAMS[0],
                VAR_P_RESOLUTION: VALUES_P_RESOLUTION[0],
                VAR_P_FPS: VALUES_P_FPS[0],
            }


        n = min(len(part) for _, part in df.groupby(PARAM_LIST))
        for configuration, data in df.groupby(PARAM_LIST):
            data = data.iloc[0:n]
            self._configuration_dfs[configuration] = cycle([data.iloc[i: i + batch_size] for i in range(0, len(data), batch_size)])


    async def request_next(self, settings):
        configuration = self._get_configuration(settings)

        return next(self._configuration_dfs[configuration])


    def _get_configuration(self, settings):
        configuration = {
            VAR_P_N_STREAMS: settings.n_streams if settings else self._default_configuration[VAR_P_N_STREAMS],
            VAR_P_FPS: settings.fps if settings else self._default_configuration[VAR_P_FPS],
            VAR_P_RESOLUTION: settings.resolution if settings else self._default_configuration[VAR_P_RESOLUTION],
        }

        return tuple(configuration[p] for p in PARAM_LIST)