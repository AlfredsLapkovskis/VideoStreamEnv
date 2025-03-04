import os
import pandas as pd
import torch as th
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import constant_fn

from learning.base_agent import BaseAgent
from learning.reinforcement_learning.environment import Environment
from learning.reinforcement_learning.hyperparams import *
from learning.reinforcement_learning.callback import RLCallback
from learning.base_hyperparams import BaseHyperParams
from utils.variables import *
from models.stream_settings import StreamSettings


class ReinforcementLearningAgent(BaseAgent[BaseHyperParams]):

    _discard_last_result = True

    def __init__(self, hyperparams, session, metric_preprocessor, logger_factory=None):
        super().__init__(hyperparams, session, metric_preprocessor, logger_factory)

        self._initialized = False
        self._finished = False
        self._last_data = None
        self._evaluation = False


    def save(self, dir, name):
        super().save(dir, name)
        return os.path.join(dir, name)
        

    def _iterate(self):        
        try:
            if not self._initialized:
                self._init_model()

                steps = 0
                while steps < self.hyperparams.total_steps:
                    train_steps = self.hyperparams.training_steps if self.hyperparams.evaluate else self.hyperparams.total_steps
                    if steps > 0 and isinstance(self._model, DQN):
                        self._model.exploration_schedule = constant_fn(self.hyperparams.final_eps)

                    self._train(train_steps)
                    steps += train_steps

                    for _ in self.hyperparams.evaluation_configurations:
                        self._evaluate()

            self._handle_methods(self._get_data())
            return True
        except StopIteration:
            return False
        

    def _train(self, steps):
        self._model.learn(steps, callback=RLCallback(self))
        

    def _evaluate(self):
        env = DummyVecEnv([lambda: self._eval_env])
        env.seed(0)
        observation = env.reset()
        states = None
        for _ in range(self.hyperparams.evaluation_steps):
            self._begin_tracking_usage()
            actions, states = self._model.predict(observation, state=states, deterministic=True)
            self._end_tracking_usage()
            observation = env.step(actions)[0]
        

    def _init_model(self):
        if self._initialized:
            return
        self._initialized = True

        self._env = Environment()
        self._env.get_initial_data = self._get_initial_data
        self._env.perform_action = self._get_perform_action()

        self._eval_env = Environment()
        self._eval_env.get_initial_data = self._get_initial_data
        self._eval_env.perform_action = self._get_perform_action()

        if isinstance(self.hyperparams, DQNHyperParams):
            model_type = DQN
            params = dict(
                policy="MultiInputPolicy",
                learning_rate=self.hyperparams.learning_rate,
                batch_size=self.hyperparams.batch_size,
                exploration_initial_eps=self.hyperparams.initial_eps,
                exploration_final_eps=self.hyperparams.final_eps,
                exploration_fraction=self.hyperparams.exploration_fraction,
                gamma=self.hyperparams.discount_factor,
                train_freq=self.hyperparams.train_freq,
                gradient_steps=self.hyperparams.gradient_steps,
                target_update_interval=self.hyperparams.target_update_interval,
                policy_kwargs=dict(
                    net_arch=self.hyperparams.net_arch,
                ),
                buffer_size=1_280_000,
            )
        elif isinstance(self.hyperparams, A2CHyperParams):
            model_type = A2C
            params = dict(
                policy="MultiInputPolicy",
                learning_rate=self.hyperparams.learning_rate,
                n_steps=self.hyperparams.batch_size,
                gamma=self.hyperparams.discount_factor,
                gae_lambda=self.hyperparams.gae_lambda,
                ent_coef=self.hyperparams.ent_coef,
                vf_coef=self.hyperparams.vf_coef,
                normalize_advantage=self.hyperparams.normalize_advantage,
                use_rms_prop=True,
                policy_kwargs=dict(
                    net_arch=self.hyperparams.net_arch,
                    optimizer_class=th.optim.RMSprop,
                ),
            )
        elif isinstance(self.hyperparams, PPOHyperParams):
            model_type = PPO
            params = dict(
                policy="MultiInputPolicy",
                learning_rate=self.hyperparams.learning_rate,
                n_steps=self.hyperparams.n_steps,
                gamma=self.hyperparams.discount_factor,
                gae_lambda=self.hyperparams.gae_lambda,
                clip_range=self.hyperparams.clip_range,
                vf_coef=self.hyperparams.vf_coef,
                ent_coef=self.hyperparams.ent_coef,
                normalize_advantage=self.hyperparams.normalize_advantage,
                batch_size=self.hyperparams.batch_size,
                n_epochs=self.hyperparams.n_epochs,
                policy_kwargs=dict(
                    net_arch=self.hyperparams.net_arch,
                ),
            )
        else:
            raise ValueError("Unexpected type of hyperparams.")

        if self.hyperparams.model_path:
            if model_type == DQN:
                params["exploration_fraction"] = 0.0

            print(f"Load model from path={self.hyperparams.model_path}")
            base_path = self.hyperparams.model_path
            
            self._model = model_type.load(
                path=os.path.join(base_path, "model.zip"),
                env=self._env,
                device="cpu",
                custom_objects=params,
                seed=0,
            )
            if isinstance(self._model, OffPolicyAlgorithm):
                self._model.load_replay_buffer(
                    path=os.path.join(base_path, "replay_buffer"),
                    truncate_last_traj=False,
                )
        else:
            self._model = model_type(
                env=self._env,
                device="cpu",
                **params,
                seed=0,
            )
        
    
    def _get_perform_action(self):
        def perform(action):
            self._suggest_settings(StreamSettings(
                id=0,
                n_streams=action[VAR_P_N_STREAMS],
                fps=action[VAR_P_FPS],
                resolution=action[VAR_P_RESOLUTION],
            ))
            while True:
                data = self._get_data()
                if isinstance(data, pd.DataFrame):
                    self._last_data = data
                    return data
                else:
                    self._handle_methods(data)
                    continue
            
        return perform
    

    def _get_initial_data(self):
        while True:
            data = self._get_data()
            if isinstance(data, pd.DataFrame):
                return data
            else:
                self._handle_methods(data)
    

    def _handle_methods(self, data):
        if isinstance(data, tuple):
            method, args = data
            if method == "save":
                dir, name = args
                self._save(dir, name)
            elif method == "eval":
                self._evaluation = True
                self._usage_logger.set_evaluation(True, new_round=True)
            elif method == "end_eval":
                self._evaluation = False
                self._usage_logger.set_evaluation(False)
        else:
            raise StopIteration()
        

    def _save(self, dir, name):
        content_dir = os.path.join(dir, name)
        os.makedirs(content_dir, exist_ok=True)

        model_path = os.path.join(content_dir, "model.zip")
        replay_buffer_path = os.path.join(content_dir, "replay_buffer")

        self._model.save(model_path)
        if isinstance(self._model, OffPolicyAlgorithm):
            self._model.save_replay_buffer(replay_buffer_path)
