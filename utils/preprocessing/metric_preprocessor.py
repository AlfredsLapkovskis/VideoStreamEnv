import pandas as pd
from abc import ABC, abstractmethod

from models.session import Session
from utils.variables import *


class MetricPreprocessor(ABC):
    def __init__(
        self,
        session: Session
    ):
        self.session = session


    @abstractmethod
    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass
