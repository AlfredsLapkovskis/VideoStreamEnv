from dataclasses import dataclass, field

from utils.variables import *


@dataclass
class BaseHyperParams:
    model_path: str = None
    total_steps: int = 1_280_000
    evaluate: bool = False
    training_steps: int = 6400
    evaluation_steps: int = 640
    evaluation_configurations: list = field(default_factory=lambda: [
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[0],  VAR_P_N_STREAMS: VALUES_P_N_STREAMS[0],  VAR_P_FPS: VALUES_P_FPS[0]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[0],  VAR_P_N_STREAMS: VALUES_P_N_STREAMS[0],  VAR_P_FPS: VALUES_P_FPS[-1]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[0],  VAR_P_N_STREAMS: VALUES_P_N_STREAMS[-1], VAR_P_FPS: VALUES_P_FPS[0]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[0],  VAR_P_N_STREAMS: VALUES_P_N_STREAMS[-1], VAR_P_FPS: VALUES_P_FPS[-1]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[-1], VAR_P_N_STREAMS: VALUES_P_N_STREAMS[0],  VAR_P_FPS: VALUES_P_FPS[0]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[-1], VAR_P_N_STREAMS: VALUES_P_N_STREAMS[0],  VAR_P_FPS: VALUES_P_FPS[-1]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[-1], VAR_P_N_STREAMS: VALUES_P_N_STREAMS[-1], VAR_P_FPS: VALUES_P_FPS[0]},
        {VAR_P_RESOLUTION: VALUES_P_RESOLUTION[-1], VAR_P_N_STREAMS: VALUES_P_N_STREAMS[-1], VAR_P_FPS: VALUES_P_FPS[-1]}
    ])
    data_batch_size: int = 1
