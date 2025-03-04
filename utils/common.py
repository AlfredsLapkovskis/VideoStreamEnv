import os
import sys
import pandas as pd
import numpy as np
import random

from models.metrics import Metrics
from models.session import Session
from utils.variables import *


def metrics_to_data_frame(metrics_batch: list[Metrics], session: Session) -> pd.DataFrame:
    df = pd.DataFrame([vars(m) for m in metrics_batch])
    setting_ids = df["setting_id"]
    settings_dict = {
        id: session.get_stream_settings(id)
        for id in set(setting_ids)
    }
    df[VAR_P_RESOLUTION] = [settings_dict[id].resolution for id in setting_ids]
    df[VAR_P_N_STREAMS] = [settings_dict[id].n_streams for id in setting_ids]
    df[VAR_P_FPS] = [settings_dict[id].fps for id in setting_ids]
    return df


def ensure_deterministic_execution():
    random.seed(0)
    np.random.seed(0)
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def save_model(agent, session, experiment, prefix="exp"):
    dir = os.path.join(session.resource_dir, "models", session.datetime_string)
    os.makedirs(dir, exist_ok=True)
    path = agent.save(dir, f"{prefix}{experiment.index}")
    print(f"Saved model at {path=}")


def action_values_to_indices(values):
    return [
        _ACTION_VALUES_TO_INDICES[action][value]
        for action, value
        in zip(PARAM_LIST, values)
    ]
    

def action_indices_to_values(indices):
    return [
        VALUES_P_MAP[action][index]
        for action, index
        in zip(PARAM_LIST, indices)
    ]


_ACTION_VALUES_TO_INDICES = {
    action: { v: i for i, v in enumerate(values) }
    for action, values
    in VALUES_P_MAP.items()
}
