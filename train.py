import asyncio
import argparse
import pandas as pd
import warnings
import multiprocessing

from models.session import Session
from models.messages import *
from learning.agent import Agent
from utils.preprocessing.aif_metric_preprocessor import AIFMetricPreprocessor
from utils.preprocessing.rl_metric_preprocessor import RLMetricPreprocessor
from utils.preprocessing.heating_metric_preprocessor_wrapper import HeatingMetricPreprocessorWrapper
from utils.variables import *
from utils.logging.tf_scalar_logger_factory import TFScalarLoggerFactory
from utils.experiment import Experiment
from utils.common import ensure_deterministic_execution, save_model
from utils.data_iterators.data_frame_data_iterator import DataFrameDataIterator


warnings.filterwarnings(action="ignore")
ensure_deterministic_execution()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        type=int,
        required=True,
        help="Index of experiment (e.g., 'experiments/exp<index>.json')",
    )
    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Path to a csv file with pre-collected metrics",
    )
    parser.add_argument(
        "-s",
        type=bool,
        required=False,
        default=False,
        help="Whether to save a model",
    )

    args = parser.parse_args()
    experiment_number = args.e
    data_path = args.d
    save = args.s
    
    experiment = Experiment(experiment_number)

    df = pd.read_csv(data_path)

    session = Session()
    session.slos = experiment.slos
    session.is_connected = True
    
    agent: Agent = None
    
    if experiment.preprocessor_type == AIFMetricPreprocessor:
        metric_preprocessor = AIFMetricPreprocessor(session)
    elif experiment.preprocessor_type == RLMetricPreprocessor:
        metric_preprocessor = RLMetricPreprocessor(session, df)
    else:
        raise ValueError("Unexpected experiment.preprocessor_type")
    
    if experiment.heating:
        metric_preprocessor = HeatingMetricPreprocessorWrapper(metric_preprocessor, k=experiment.heating_k)

    agent = experiment.agent_type(
        hyperparams=experiment.hparams,
        session=session,
        metric_preprocessor=metric_preprocessor,
        logger_factory=TFScalarLoggerFactory(f"exp{experiment_number}", minimal=False),
    )

    try:
        print("Begin training")
        await agent.fit(
            train_iterator=DataFrameDataIterator(
                df=df,
                batch_size=experiment.hparams.data_batch_size,
            ),
            eval_iterator_factory=lambda configuration: DataFrameDataIterator(
                df=df,
                batch_size=experiment.hparams.data_batch_size,
                default_configuration=configuration,
            ),
        )

        if save:
            save_model(agent, session, experiment)
    except Exception as exc:
        print(f"Training failed {exc=}")

    agent.stop(wait=True)
    print("Finished")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    asyncio.run(main())
