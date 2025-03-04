from abc import ABC, abstractmethod
import pandas as pd


class ScalarLogger(ABC):
    @abstractmethod
    def log_from_batch(self, df: pd.DataFrame):
        pass
    @abstractmethod
    def log_usage(self, cpu_time, memory):
        pass
    @abstractmethod
    def set_evaluation(self, evaluation, new_round=False):
        pass
