import os
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import groupby

from utils.logging.scalar_logger import ScalarLogger
from models.session import Session
from models.slo_fulfillment import SLOFulfillment
from utils.variables import *


class TFScalarLogger(ScalarLogger):

    def __init__(
        self,
        session: Session,
        name="",
        minimal=False,
    ):
        self.session = session
        self.minimal = minimal
        
        path_components = [session.logs_dir, session.datetime_string]
        if name:
            path_components.append(name)

        log_dir = os.path.join(*path_components)
        self._file_writer = tf.summary.create_file_writer(log_dir)
        # We assume that log_from_batch and log_usage won't be executed from the same process,
        # and it is a caller's responsibility to synchronize these function calls to ensure consistent _step values.
        self._steps = {}
        self._is_evaluation = False
        self._evaluation_slo_fulfillments = [[]]
        self._evaluation_cpu = [[]]
        self._evaluation_memory = [[]]
        self._round = 0
        

    def log_from_batch(self, df: pd.DataFrame):
        mean_series = df.mean()
        mean_series["thermal_state"] = df["thermal_state"].max()
        sf = SLOFulfillment.calc(mean_series, self.session.slos)

        if self._is_evaluation:
            self._evaluation_slo_fulfillments[self._round].append(sf)
        else:
            self._log_from_batch(sf, prefix="")


    def log_usage(self, cpu_time, memory):
        if self._is_evaluation:
            self._evaluation_cpu[self._round].append(cpu_time)
            self._evaluation_memory[self._round].append(memory)
        else:
            self._log_usage(cpu_time, memory, prefix="")


    def set_evaluation(self, evaluation, new_round=False):
        if evaluation and self._is_evaluation and new_round:
            self._evaluation_slo_fulfillments.append([])
            self._evaluation_cpu.append([])
            self._evaluation_memory.append([])
            self._round += 1
        elif self._is_evaluation == evaluation:
            return
        self._is_evaluation = evaluation
        if evaluation:
            return
        
        def all_equal(arrays):
            it = groupby(arrays, len)
            return next(it, True) and not next(it, False)

        if self._evaluation_slo_fulfillments != [[]]:
            assert all_equal(self._evaluation_slo_fulfillments), f"{self._evaluation_slo_fulfillments=}"

            for step in range(len(self._evaluation_slo_fulfillments[0])):
                sfs = [self._evaluation_slo_fulfillments[r][step] for r in range(self._round + 1)]
                mean_sf, std_sf = SLOFulfillment.calc_stats(sfs)

                self._log_from_batch(mean_sf, "eval_mean")
                self._log_from_batch(std_sf, "eval_std")

            self._evaluation_slo_fulfillments = [[]]

        if self._evaluation_cpu != [[]] and self._evaluation_memory != [[]]:
            assert all_equal(self._evaluation_cpu) and all_equal(self._evaluation_memory), f"{self._evaluation_cpu=}, {self._evaluation_memory=}"
            assert len(self._evaluation_cpu) == len(self._evaluation_memory), f"{self._evaluation_cpu=}, {self._evaluation_memory=}"

            for step in range(len(self._evaluation_cpu[0])):
                cpu_time_list = [self._evaluation_cpu[r][step] for r in range(self._round + 1)]
                memory_list = [self._evaluation_memory[r][step] for r in range(self._round + 1)]
                memory_list = [m for m in memory_list if m > 0]

                mean_cpu_time, std_cpu_time = np.mean(cpu_time_list), np.std(cpu_time_list)
                mean_memory, std_memory = (np.mean(memory_list), np.std(memory_list)) if memory_list else (-1., -1.)

                self._log_usage(mean_cpu_time, mean_memory, "eval_mean")
                self._log_usage(std_cpu_time, std_memory, "eval_std")

            self._evaluation_cpu = [[]]
            self._evaluation_memory = [[]]

        self._round = 0


    def _log_from_batch(self, sf: SLOFulfillment, prefix):
        step = self._steps.get(prefix, 0)

        with self._file_writer.as_default() as _:
            tf.summary.scalar(self._name("SLO_ALL", prefix), sf.slo_fulfillment, step)
            if not self.minimal:
                tf.summary.scalar(self._name("SLO_FPS", prefix), sf.avg_actual_fps, step)
                tf.summary.scalar(self._name("SLO_network", prefix), sf.network_usage, step)
                tf.summary.scalar(self._name("SLO_thermal_state", prefix), sf.thermal_state, step)
                tf.summary.scalar(self._name("SLO_streams", prefix), sf.stream_fulfillment, step)
                tf.summary.scalar(self._name("SLO_render_scale", prefix), sf.avg_render_scale_factor, step)
                tf.summary.scalar(self._name("SLO_QoS", prefix), sf.qos, step)
                tf.summary.scalar(self._name("SLO_QoE", prefix), sf.qoe, step)

            self._steps[prefix] = step + 1


    def _log_usage(self, cpu_time, memory, prefix):
        step = self._steps.get(prefix, 0)

        with self._file_writer.as_default() as _:
            tf.summary.scalar(self._name("CPU_time", prefix), cpu_time / 1000 / 1000, step) # milliseconds
            tf.summary.scalar(self._name("Memory", prefix), memory / 1024 / 1024, step) # megabytes

            self._steps[prefix] = step + 1


    def _name(self, scalar_name, prefix):
        return (f"{prefix}_" if prefix else "") + scalar_name
