import asyncio
import websockets
from websockets.asyncio.server import ServerConnection
import warnings
import argparse
import pandas as pd

from models.session import Session
from models.messages import *
from learning.base_agent import BaseAgent
from utils.message_coder import MessageCoder
from utils.preprocessing.aif_metric_preprocessor import AIFMetricPreprocessor
from utils.preprocessing.rl_metric_preprocessor import RLMetricPreprocessor
from utils.preprocessing.heating_metric_preprocessor_wrapper import HeatingMetricPreprocessorWrapper
from services.message_receiver import MessageReceiver
from services.message_sender_with_rate_limit import MessageSenderWithRateLimit
from services.video_streamer import VideoStreamer
from utils.experiment import Experiment
from utils.common import ensure_deterministic_execution, save_model
from utils.logging.tf_scalar_logger_factory import TFScalarLoggerFactory
from utils.data_iterators.real_time_data_iterator import RealTimeDataIterator


warnings.filterwarnings(action="once")
ensure_deterministic_execution()


HOST = "0.0.0.0"
PORT = 8888


message_coder = MessageCoder()


async def handler(args, connection: ServerConnection):
    experiment = Experiment(args.e)
    training_data = pd.read_csv(args.d) if args.d else None
    save = args.s
    rate_limit = args.r # May be None, i.e., no rate limit applied

    session = Session()
    message_sender = MessageSenderWithRateLimit(connection, message_coder, rate_limit=rate_limit)
    video_streamer = VideoStreamer(session, message_sender)
    iterator = RealTimeDataIterator(message_sender)

    logger_factory = TFScalarLoggerFactory(f"server{experiment.index}")

    if experiment.preprocessor_type == AIFMetricPreprocessor:
        metric_preprocessor = AIFMetricPreprocessor(session)
    elif experiment.preprocessor_type == RLMetricPreprocessor:
        assert(training_data is not None)
        metric_preprocessor = RLMetricPreprocessor(session, training_data)
    else:
        raise ValueError("Unexpected experiment.preprocessor_type")
    
    if experiment.heating:
        metric_preprocessor = HeatingMetricPreprocessorWrapper(metric_preprocessor, k=experiment.heating_k)

    learning_agent: BaseAgent = experiment.agent_type(
        experiment.hparams,
        session,
        metric_preprocessor,
        logger_factory,
    )
    message_receiver = MessageReceiver(
        connection, 
        session,
        message_coder, 
        message_sender, 
        video_streamer,
        learning_agent,
        iterator,
    )

    try:
        await message_receiver.receive()
    except Exception as exc:
        print(f"Connection closed with {exc=}.")
    finally:
        if save:
            save_model(learning_agent, session, experiment, "server")

        learning_agent.stop()


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
        default="",
        help="Path to pre-collected data, necessary for RLMetricPreprocessor",
    )
    parser.add_argument(
        "-s",
        type=bool,
        required=False,
        default=False,
        help="Whether to save a model",
    )
    parser.add_argument(
        "-r",
        type=int,
        required=False,
        default=None,
        help="Data transmission rate limit in bytes",
    )

    args = parser.parse_args()

    print("Running server...")
    
    server = await websockets.serve(lambda conn: handler(args, conn), HOST, PORT, ping_timeout=None)
    try:
        await server.wait_closed()
    except Exception as exc:
        print(f"Connection closed with {exc=}.")


if __name__ == "__main__":
    asyncio.run(main())
