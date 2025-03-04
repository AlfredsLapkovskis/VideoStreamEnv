from abc import abstractmethod
import asyncio
import pandas as pd
from multiprocessing import Process, Queue, current_process
from queue import Empty
from typing import Generic, TypeVar, Callable
from asyncio import CancelledError

from .agent import Agent
from learning.base_hyperparams import BaseHyperParams
from models.session import Session
from models.stream_settings import StreamSettings
from models.messages import *
from utils.preprocessing.metric_preprocessor import MetricPreprocessor
from utils.logging.scalar_logger import ScalarLogger
from utils.usage_monitor import UsageMonitor
from utils.variables import *

HP = TypeVar("HP", bound="BaseHyperParams")


class BaseAgent(Agent, Generic[HP]):

    _discard_last_result = False # Workaround required due to a reset() method in RL environment.

    def __init__(
        self,
        hyperparams: HP,
        session: Session,
        metric_preprocessor: MetricPreprocessor,
        logger_factory: Callable[[Session], ScalarLogger]=None,
    ):
        self.hyperparams = hyperparams
        self.session = session
        if current_process().name == "MainProcess":
            self.metric_preprocessor = metric_preprocessor
            self._is_stopped = False
            self._pending_data = pd.DataFrame(columns=FULL_VAR_LIST)
            self._waiting = False
            self._metric_logger = logger_factory(session) if logger_factory else None
            self._first_log_skipped = False
            self._input_queue = Queue()
            self._output_queue = Queue()
            self._subprocess = Process(target=self._process, name="VideoStreamAgent", args=(
                self.__class__,
                hyperparams,
                session,
                self._input_queue,
                self._output_queue,
                logger_factory,
            ))
            print("Start subprocess")
            self._subprocess.start()
        else:
            self._usage_monitor = UsageMonitor() if logger_factory else None
            self._usage_logger = logger_factory(session) if logger_factory else None


    async def fit(self, train_iterator, eval_iterator_factory=None) -> None:
        try:
            evaluate = self.hyperparams.evaluate
            assert not evaluate or eval_iterator_factory

            steps = 0
            while steps < self.hyperparams.total_steps:
                self._check_stopped()
                if evaluate:
                    train_steps = self.hyperparams.training_steps
                else:
                    train_steps = self.hyperparams.total_steps

                settings = None
                for i in range(train_steps + 1):
                    print(f"Training step {i}")
                    metrics = await train_iterator.request_next(settings)
                    self._check_stopped()
                    if i > 0 and self._metric_logger:
                        self._metric_logger.log_from_batch(metrics)
                    df = self.metric_preprocessor.preprocess(metrics)
                    self._input_queue.put_nowait(df)
                    if i != train_steps or not self._discard_last_result:
                        settings = await self._wait_result()
                        self._check_stopped()

                steps += train_steps

                if evaluate:
                    print("Evaluation")
                    for conf in self.hyperparams.evaluation_configurations:
                        self._input_queue.put_nowait(("eval", ()))
                        if self._metric_logger:
                            self._metric_logger.set_evaluation(True, new_round=True)
                        eval_iterator = eval_iterator_factory(conf)
                        settings = None

                        eval_steps = self.hyperparams.evaluation_steps
                        for i in range(eval_steps + 1):
                            metrics = await eval_iterator.request_next(settings)
                            self._check_stopped()
                            if i > 0:
                                self._metric_logger.log_from_batch(metrics)
                            df = self.metric_preprocessor.preprocess(metrics)
                            self._input_queue.put_nowait(df)
                            if i != eval_steps or not self._discard_last_result:
                                settings = await self._wait_result()
                                self._check_stopped()
                    self._input_queue.put_nowait(("end_eval", ()))
                    if self._metric_logger:
                        self._metric_logger.set_evaluation(False)
        except StopIteration or CancelledError as exc:
            pass


    def stop(self, wait=False) -> None:
        if self._is_stopped:
            return
        self._is_stopped = True
        
        if self._metric_logger:
            self._metric_logger.set_evaluation(False)
        self._input_queue.put(None)
        if wait:
            self._subprocess.join()


    def save(self, dir, name):
        self._input_queue.put(("save", (dir, name)))
        return None
    

    def begin_eval_round(self):
        self._first_log_skipped = False
        self._metric_logger.set_evaluation(True, new_round=True)
        self._input_queue.put_nowait(("eval", ()))

    
    def end_evaluation(self):
        self._first_log_skipped = False
        self._metric_logger.set_evaluation(False)
        self._input_queue.put_nowait(("end_eval", ()))


    def _check_stopped(self):
        if self._is_stopped:
            raise StopIteration()


    async def _wait_result(self):
        while True:
            try:
                return self._output_queue.get_nowait()
            except Empty:
                pass
            await asyncio.sleep(0.05)


    @staticmethod
    def _process(cls, hp: HP, session: Session, input_queue: Queue, output_queue: Queue, logger_factory: Callable[[Session], ScalarLogger]):
        self: BaseAgent = cls(hp, session, None, logger_factory)
        self._input_queue = input_queue
        self._output_queue = output_queue

        while True:
            if not self._iterate():
                break
        
        if self._usage_logger:
            self._usage_logger.set_evaluation(False)
        if self._usage_monitor:
            self._usage_monitor.close(wait=True)


    @abstractmethod
    def _iterate(self) -> bool:
        pass


    def _get_data(self):
        assert(current_process().name != "MainProcess")

        return self._input_queue.get()


    def _suggest_settings(self, settings: StreamSettings):
        print(f"Suggest {settings=}")
        self._output_queue.put_nowait(settings)

    
    def _begin_tracking_usage(self):
        if self._usage_monitor and self._usage_logger:
            self._usage_monitor.start()


    def _end_tracking_usage(self):
        if self._usage_monitor and self._usage_logger:
            cpu_time, memory = self._usage_monitor.stop()
            self._usage_logger.log_usage(cpu_time, memory)
