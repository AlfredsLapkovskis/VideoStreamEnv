import numpy as np
import pandas as pd
import gymnasium as gym
from functools import reduce
from itertools import product

from utils.variables import *
from utils.common import action_values_to_indices


class Environment(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    _ACTIONS = list(product(*[VALUES_P_MAP[p] for p in PARAM_LIST]))

    def __init__(self):
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(len(self._ACTIONS))
        self.observation_space = gym.spaces.Dict({
            **{p: gym.spaces.Discrete(len(VALUES_P_MAP[p])) for p in PARAM_LIST},
            VAR_M_CPU_USAGE: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            VAR_M_MEMORY_USAGE: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            VAR_S_AVG_ACTUAL_FPS: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            VAR_S_NETWORK_USAGE: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            VAR_S_AVG_RENDER_SCALE_FACTOR: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            VAR_S_THERMAL_STATE: gym.spaces.Discrete(4),
        })
        self.reward_range = (0., 1.)


    def step(self, action):
        assert(hasattr(self, "perform_action"))

        action_values = self._ACTIONS[action]

        data = self.perform_action({
            key: value
            for key, value in zip(PARAM_LIST, action_values)
        })

        indexer = reduce(
            lambda acc, x: acc & (data[x[0]] == x[1]),
            zip(PARAM_LIST, action_values),
            True,
        )
        suitable_data = data[indexer]
        data_len = len(suitable_data)
        assert data_len > 0, f"{action_values=}"

        observation = self._data_to_observation(suitable_data, action_values)
        print(f"{observation=}")

        reward = suitable_data["reward"].mean()
        print(f"{reward=}")

        return observation, reward, False, False, {}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        assert(hasattr(self, "get_initial_data"))

        data: pd.DataFrame = self.get_initial_data()
        assert(len(data) > 0)
        last = data.iloc[-1]
        action_values = [last[p] for p in PARAM_LIST]
        indexer = reduce(
            lambda acc, x: acc & (data[x[0]] == x[1]),
            zip(PARAM_LIST, action_values),
            True,
        )
        suitable_data = data[indexer]
        observation = self._data_to_observation(suitable_data, action_values)

        return observation, {}
    

    def render(self):
        pass


    def close(self):
        pass


    def _data_to_observation(self, data, action_values):
        return {
            **dict(zip(PARAM_LIST, action_values_to_indices(action_values))),
            VAR_M_CPU_USAGE: data[VAR_M_CPU_USAGE].mean(),
            VAR_M_MEMORY_USAGE: data[VAR_M_MEMORY_USAGE].mean(),
            VAR_S_AVG_ACTUAL_FPS: data[VAR_S_AVG_ACTUAL_FPS].mean(),
            VAR_S_NETWORK_USAGE: data[VAR_S_NETWORK_USAGE].mean(),
            VAR_S_AVG_RENDER_SCALE_FACTOR: data[VAR_S_AVG_RENDER_SCALE_FACTOR].mean(),
            VAR_S_THERMAL_STATE: data[VAR_S_THERMAL_STATE].max(),
        }
