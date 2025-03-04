import json
import os

from models.service_level_objectives import ServiceLevelObjectives
from learning.active_inference.active_inference_agent import ActiveInferenceAgent
from learning.reinforcement_learning.reinforcement_learning_agent import ReinforcementLearningAgent
from learning.random.random_agent import RandomAgent
from learning.sequential.sequential_agent import SequentialAgent
from learning.base_hyperparams import BaseHyperParams
from learning.active_inference.hyperparams import HyperParams as HPActiveInference
from learning.reinforcement_learning.hyperparams import DQNHyperParams, A2CHyperParams, PPOHyperParams
from utils.preprocessing.aif_metric_preprocessor import AIFMetricPreprocessor
from utils.preprocessing.rl_metric_preprocessor import RLMetricPreprocessor


class Experiment:
    _AGENT_MAP = {
        "AIF": ActiveInferenceAgent,
        "DQN": ReinforcementLearningAgent,
        "A2C": ReinforcementLearningAgent,
        "PPO": ReinforcementLearningAgent,
        "RAND": RandomAgent,
        "SEQ": SequentialAgent,
    }
    _HP_MAP = {
        "AIF": HPActiveInference,
        "DQN": DQNHyperParams,
        "A2C": A2CHyperParams,
        "PPO": PPOHyperParams,
        "RAND": BaseHyperParams,
        "SEQ": BaseHyperParams,
    }
    _PREPROCESSOR_MAP = {
        "AIF": AIFMetricPreprocessor,
        "DQN": RLMetricPreprocessor,
        "A2C": RLMetricPreprocessor,
        "PPO": RLMetricPreprocessor,
        "RAND": AIFMetricPreprocessor,
        "SEQ": AIFMetricPreprocessor,
    }
    
    def __init__(self, index):
        with open(os.path.join("experiments", f"exp{index}.json"), "r") as f:
            experiment = json.loads(f.read())
        self.index = index
        self.type = experiment["type"]
        self.agent_type = self._AGENT_MAP[self.type]
        self.preprocessor_type = self._PREPROCESSOR_MAP[self.type]
        self.hparams: BaseHyperParams = self._HP_MAP[self.type](**(experiment["hparams"] if "hparams" in experiment else {}))

        self.slos = None
        if "slos" in experiment:
            self.slos = ServiceLevelObjectives(**experiment["slos"])

        self.heating = False
        self.heating_k = None
        if "extras" in experiment:
            extras = experiment["extras"]

            self.heating = extras.get("heating", False)
            self.heating_k = extras.get("heating_k", None)
