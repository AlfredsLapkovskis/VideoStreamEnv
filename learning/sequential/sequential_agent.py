import pandas as pd
from learning.base_agent import BaseAgent
from learning.base_hyperparams import BaseHyperParams
from utils.variables import *
from models.stream_settings import StreamSettings

from itertools import product


class SequentialAgent(BaseAgent[BaseHyperParams]):

    def __init__(self, hyperparams, session, message_sender, metric_preprocessor):
        super().__init__(hyperparams, session, message_sender, metric_preprocessor)

        self._index = 0
        self._started = False
        self._possible_actions = list(product(*[VALUES_P_MAP[p] for p in PARAM_LIST]))
        self._reversed_possible_actions = list(reversed(self._possible_actions))


    def _iterate(self):
        data = self._get_data()
        if data is None:
            return False
        elif not isinstance(data, pd.DataFrame):
            return True
        
        if not data.empty and not self._started:
            prev_configuration = tuple([data.iloc[-1][p] for p in PARAM_LIST])
            self._index = self._possible_actions.index(prev_configuration)
        self._started = True

        actions_len = len(self._possible_actions)
        reverse_direction = ((self._index // actions_len) & 1) == 1
        actual_index = self._index % actions_len

        action = self._possible_actions[actual_index] if not reverse_direction else self._reversed_possible_actions[actual_index]

        self._index += 1

        settings = StreamSettings(
            id=0,
            n_streams=action[PARAM_LIST.index(VAR_P_N_STREAMS)],
            fps=action[PARAM_LIST.index(VAR_P_FPS)],
            resolution=action[PARAM_LIST.index(VAR_P_RESOLUTION)],
        )
        print(settings)
        self._suggest_settings(settings)

        return True
