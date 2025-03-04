from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable

from utils.data_iterators.data_iterator import DataIterator


class Agent(ABC):
    @abstractmethod
    async def fit(self, train_iterator: DataIterator, eval_iterator_factory: Callable[[dict], DataIterator]=None) -> None:
        pass
    @abstractmethod
    def stop(self, wait=False) -> None:
        pass
    @abstractmethod
    def save(self, dir, name) -> str:
        pass
