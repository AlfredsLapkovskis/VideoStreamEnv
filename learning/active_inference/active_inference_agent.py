import os
import numpy as np
import pandas as pd
import json
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, AICScore
from pgmpy.inference import VariableElimination
from copy import deepcopy
from itertools import product, chain
from scipy.interpolate import griddata
from pgmpy.factors.discrete import DiscreteFactor

from models.stream_settings import StreamSettings
from learning.base_agent import BaseAgent
from learning.active_inference.hyperparams import HyperParams
from utils.variables import *
from utils.common import action_values_to_indices


class ActiveInferenceAgent(BaseAgent[HyperParams]):

    _discard_last_result = False

    def __init__(self, hyperparams, session, metric_preprocessor, logger_factory=None):
        super().__init__(hyperparams, session, metric_preprocessor, logger_factory)

        self._model: BayesianNetwork | None = None
        self._dataset = pd.DataFrame(columns=FULL_VAR_LIST)
        self._surprise_buffer = []
        self._configuration_surprises = dict()
        self._rebuild_counter = 0
        self._retrain_counter = 0
        self._evaluation = False
        self._new_eval_round = False

        self._pv_tensor = np.full(VALUES_P_SHAPE, -1.0)
        self._ra_tensor = np.full(VALUES_P_SHAPE, -1.0)
        self._ig_tensor = np.full(VALUES_P_SHAPE, -1.0)
        self._e_tensor = np.full(VALUES_P_SHAPE, 0.0)

        for indices_and_e in hyperparams.initial_additional_surprises:
            self._e_tensor[*indices_and_e[:-1]] = indices_and_e[-1]

        if self.hyperparams.model_path:
            print(f"Load model from path={self.hyperparams.model_path}")
            
            base_path = self.hyperparams.model_path

            self._model = BayesianNetwork.load(
                os.path.join(base_path, "model.bif"),
                state_name_type=int,
            )
            self._dataset = pd.read_csv(os.path.join(base_path, "data.csv"))

            with open(os.path.join(base_path, "values.json"), "r") as f:
                values = json.loads(f.read())
                self._surprise_buffer = values["surprise_buffer"]
                self._configuration_surprises = {
                    tuple([int(c) for c in s.split("_")]): arr 
                    for s, arr in values["configuration_surprises"].items()
                }
                self._pv_tensor = np.array(values["pv"]).reshape(VALUES_P_SHAPE)
                self._ra_tensor = np.array(values["ra"]).reshape(VALUES_P_SHAPE)
                self._ig_tensor = np.array(values["ig"]).reshape(VALUES_P_SHAPE)
                self._e_tensor = np.array(values["e"]).reshape(VALUES_P_SHAPE)


    def save(self, dir, name):
        super().save(dir, name)
        return os.path.join(dir, name)


    def _iterate(self):
        df = self._get_data()
        if isinstance(df, pd.DataFrame):
            self._begin_tracking_usage()
            self._fit(df)
            self._end_tracking_usage()
            return True
        elif isinstance(df, tuple):
            method, args = df
            if method == "save":
                dir, name = args
                self._save(dir, name)
                return True
            elif method == "eval":
                self._evaluation = True
                self._new_eval_round = True
                self._usage_logger.set_evaluation(self._evaluation, self._new_eval_round)
                return True
            elif method == "end_eval":
                self._evaluation = False
                self._new_eval_round = False
                self._usage_logger.set_evaluation(False)
                return True
        return False
    

    def _save(self, dir, name):
        if not self._model:
            return
        content_dir = os.path.join(dir, name)
        os.makedirs(content_dir, exist_ok=True)

        model_path = os.path.join(content_dir, "model.bif")
        data_path = os.path.join(content_dir, "data.csv")
        values_path = os.path.join(content_dir, "values.json")

        self._model.save(model_path)
        self._dataset.to_csv(data_path, index=False)
        
        with open(values_path, "w") as f:
            f.write(json.dumps({
                "surprise_buffer": self._surprise_buffer,
                "configuration_surprises": {
                    "_".join([str(c) for c in configuration]): arr 
                    for configuration, arr in self._configuration_surprises.items()
                },
                "pv": list(self._pv_tensor.flatten()),
                "ra": list(self._ra_tensor.flatten()),
                "ig": list(self._ig_tensor.flatten()),
                "e": list(self._e_tensor.flatten()),
            }, indent=2))


    def _fit(self, df: pd.DataFrame):
        if not self._evaluation:
            self._dataset = pd.concat([self._dataset, df])
            max_size = self.hyperparams.data_batch_size * self.hyperparams.total_steps
            if len(self._dataset) > max_size:
                self._dataset = self._dataset.iloc[-max_size:]

        if not self._model:
           self._init_model()
        
        if not self._evaluation:
            if self._are_all_states_known(df):
                configuration_surprises, surprise = self._calc_surprises(df)
                for configuration, c_surprises in configuration_surprises.items():
                    if configuration in self._configuration_surprises:
                        self._configuration_surprises[configuration].extend(c_surprises)
                    else:
                        self._configuration_surprises[configuration] = c_surprises

                self._add_surprise(surprise)

                median_surprise = np.median(self._surprise_buffer)

                if surprise > (median_surprise * self.hyperparams.surprise_threshold_factor):
                    self._rebuild_counter += 1
                    self._init_model()
                elif surprise > median_surprise:
                    self._retrain_counter += 1
                    self._retrain(df)
            else:
                self._retrain(df)

        self._calc_factors()
        settings = self._infer_settings()

        print(settings)

        self._suggest_settings(settings)
        

    def _init_model(self):
        dag = HillClimbSearch(self._dataset).estimate(
            scoring_method=AICScore(self._dataset),
            max_indegree=self.hyperparams.graph_max_indegree,
            epsilon=self.hyperparams.hill_climb_epsilon,
        )
        self._model = BayesianNetwork(dag)
        self._model.fit(
            data=self._dataset,
            estimator=MaximumLikelihoodEstimator,
        )


    def _retrain(self, df: pd.DataFrame):
        n_prev_samples = (len(self._dataset) - len(df)) * self.hyperparams.weight_of_past_data

        try:
            self._model.fit_update(df, n_prev_samples=n_prev_samples)
        except ValueError:
            self._model.fit(self._dataset)


    def _are_all_states_known(self, df):
        for var in FULL_VAR_LIST:
            for _, row in df.iterrows():
                if row[var] not in self._model.states[var]:
                    return False

            for v in self._model.get_markov_blanket(var):
                for _, row in df.iterrows():
                    if row[v] not in self._model.states[v]:
                        return False

        return True


    def _calc_factors(self):
        assert(self._configuration_surprises)
        surprises = list(chain(*self._configuration_surprises.values()))

        max_surprise = np.max(surprises)
        mean_surprise = np.mean(surprises)

        self._ig_tensor = np.full(VALUES_P_SHAPE, max_surprise) + self._e_tensor

        inference = VariableElimination(self._get_markov_blanket_of_vars([*SLO_LIST, *PARAM_LIST]))

        known_params = [self._model.states[p] for p in PARAM_LIST]
        param_combinations = product(*known_params)

        for combination, surprises in self._configuration_surprises.items():
            indices = action_values_to_indices(combination)
            ig = np.median(surprises) / mean_surprise
            self._ig_tensor[*indices] = ig if not np.isnan(ig) else 0.0
            self._e_tensor[*indices] = 0.0
        
        for combination in param_combinations:
            evidence = dict(zip(PARAM_LIST, combination))
            
            # Returns expected mean of the variables mapped to real interval [0, 1], 
            # i.e. E[(f(x_1), ..., f(x_n)) / n], where f(x) = min(1.0, 0.2 * x + 0.1)
            def calc_value(result: DiscreteFactor, vars: list[str]):
                indexed_state_names = [
                    [(i, state) for i, state in enumerate(result.state_names[var])]
                    for var in vars
                ]

                acc = 0.
                for indexed_states in product(*indexed_state_names):
                    indices = [x[0] for x in indexed_states]
                    states = [x[1] for x in indexed_states]

                    probability = result.values[*indices]
                    # Converting a discrete value to an average probability in a continuous interval that it represents.
                    acc += sum(min(1., 0.2 * s + 0.1) for s in states) * probability

                return acc / len(vars)

            qoe_result = inference.query(QOE_SLO_LIST, evidence)
            qos_result = inference.query(QOS_SLO_LIST, evidence)

            pv = calc_value(qoe_result, QOE_SLO_LIST)
            ra = calc_value(qos_result, QOS_SLO_LIST)

            if np.isnan(pv) or np.isnan(ra):
                continue

            indices = action_values_to_indices(combination)
            self._pv_tensor[*indices] = pv
            self._ra_tensor[*indices] = ra
    

    def _infer_settings(self):
        pv_tensor = self._interpolate_tensor(self._pv_tensor)
        ra_tensor = self._interpolate_tensor(self._ra_tensor)
        ig_tensor = self._ig_tensor

        max_u = -float("inf")
        best_combination = None

        log_str = ""

        for combination in product(*[VALUES_P_MAP[var] for var in PARAM_LIST]):
            indices = action_values_to_indices(combination)
            pv = pv_tensor[*indices]
            ra = ra_tensor[*indices]
            ig = ig_tensor[*indices] if not self._evaluation else 0.
            u = pv + ra + ig
            if u > max_u:
                max_u = u
                best_combination = combination
                log_str = f"Best values: {u=}, {pv=}, {ra=}, {ig=}"

        print(log_str)

        return StreamSettings(
            id=0,
            n_streams=best_combination[PARAM_LIST.index(VAR_P_N_STREAMS)],
            fps=best_combination[PARAM_LIST.index(VAR_P_FPS)],
            resolution=best_combination[PARAM_LIST.index(VAR_P_RESOLUTION)],
        )    


    def _interpolate_tensor(self, tensor: np.array, method="linear", _repeated=False):
        if len(tensor.shape) >= 2 and not _repeated:
            # griddata returns a transposed tensor
            tensor = tensor.transpose((1, 0, *range(2, len(tensor.shape))))
        xs = [np.arange(s) for s in tensor.shape]
        xxs = tuple(np.meshgrid(*xs))

        xxs_flat = [xx.flatten() for xx in xxs]

        tensor_flat = tensor.flatten()

        filled_indices = tensor_flat != -1
        xxs_filled = tuple(xx[filled_indices] for xx in xxs_flat)

        tensor_filled = tensor_flat[filled_indices]

        try:
            interpolated_tensor = griddata(xxs_filled, tensor_filled, xxs, method)

            mask = np.isnan(interpolated_tensor)
            interpolated_tensor[mask] = griddata(xxs_filled, tensor_filled, tuple(xx[mask] for xx in xxs), "nearest")

            return interpolated_tensor
        except Exception:
            return self._interpolate_tensor(tensor, "nearest", _repeated=True)


    def _calc_surprises(self, df: pd.DataFrame):
        inference = VariableElimination(self._get_markov_blanket_of_vars(SLO_LIST))

        configuration_surprises = dict()
        total_surprise = 0

        for var in SLO_LIST:
            log_likelihood = 0
            evidence_vars = self._model.get_markov_blanket(var)

            for _, row in df.iterrows():
                evidence = {var: row[var] for var in evidence_vars}
                result = inference.query([var], evidence)
                index = result.state_names[var].index(row[var])
                p = result.values[index]
                surprise = -np.log(p if p > 0 else 1e-10)
                log_likelihood -= surprise

                configuration = tuple(row[PARAM_LIST])
                if configuration in configuration_surprises:
                    configuration_surprises[configuration].append(surprise)
                else:
                    configuration_surprises[configuration] = [surprise]
            
            cpd = self._get_markov_blanket_of_vars([var]).get_cpds(var)

            k = len(cpd.get_values().flatten()) - len(cpd.variables)
            n = len(df)

            bic = -2 * log_likelihood + k * np.log(n)

            total_surprise += bic

        configuration_surprises = {
            configuration: [sum(surprises) / len(surprises)] # TODO
            for configuration, surprises in configuration_surprises.items()
        }

        return configuration_surprises, total_surprise


    def _get_markov_blanket_of_vars(self, vars):
        nodes = set(vars)
        for var in vars:
            nodes.update(self._model.get_markov_blanket(var))
        
        assert(self._model)
        mb = deepcopy(self._model)

        for node in self._model.nodes:
            if node not in nodes:
                mb.remove_node(node)

        return mb
    

    def _add_surprise(self, surprise):
        if len(self._surprise_buffer) >= self.hyperparams.surprise_buffer_size:
            self._surprise_buffer.pop(0)
        self._surprise_buffer.append(surprise)
