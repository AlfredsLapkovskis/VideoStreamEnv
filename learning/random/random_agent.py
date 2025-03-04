import pandas as pd
from learning.base_agent import BaseAgent
from learning.base_hyperparams import BaseHyperParams
from utils.variables import *
from models.stream_settings import StreamSettings

from random import shuffle
from itertools import product


class RandomAgent(BaseAgent[BaseHyperParams]):

    def __init__(self, hyperparams, session, message_sender, metric_preprocessor):
        super().__init__(hyperparams, session, message_sender, metric_preprocessor)

        self._index = 0
        self._possible_actions = list(product(*[VALUES_P_MAP[p] for p in PARAM_LIST]))
        shuffle(self._possible_actions)


    def _iterate(self):
        data = self._get_data()
        if data is None:
            return False
        elif not isinstance(data, pd.DataFrame):
            return True

        if self._index == len(self._possible_actions):
            self._index = 0
            shuffle(self._possible_actions)

        action = self._possible_actions[self._index]
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
