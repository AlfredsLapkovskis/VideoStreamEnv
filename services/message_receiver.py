import asyncio
from websockets.asyncio.server import ServerConnection
import os
import pandas as pd

from models.session import Session
from models.messages import *
from utils.message_coder import MessageCoder
from utils.common import metrics_to_data_frame
from utils.data_iterators.real_time_data_iterator import RealTimeDataIterator
from learning.agent import Agent
from services.message_sender import MessageSender
from services.video_streamer import VideoStreamer



class MessageReceiver:

    def __init__(
        self,
        connection: ServerConnection,
        session: Session,
        message_coder: MessageCoder,
        message_sender: MessageSender,
        video_streamer: VideoStreamer,
        learning_agent: Agent,
        data_iterator: RealTimeDataIterator,
    ):
        self.connection = connection
        self.session = session
        self.message_coder = message_coder
        self.message_sender = message_sender
        self.video_streamer = video_streamer
        self.learning_agent = learning_agent
        self.data_iterator = data_iterator

        metrics_dir = os.path.join(self.session.resource_dir, "metrics", self.session.datetime_string)
        os.makedirs(metrics_dir, exist_ok=True)

        self._metrics_path = os.path.join(metrics_dir, "metrics.csv")

    
    async def receive(self):
        print("Receiving inbound messages...")
        async for raw_message in self.connection:
            if not isinstance(raw_message, bytes):
                self.message_sender.error(OutMessageError.ERR_WRONG_MESSAGE, f"Wrong message format: {raw_message}")

            message = self.message_coder.decode_in_message(raw_message)
            if message is not None:
                self._handle_message(message)
            else:
                self.message_sender.error(OutMessageError.ERR_WRONG_MESSAGE, f"Failed to decode message: {raw_message}")

        self.video_streamer.end_streaming()
            
    
    def _handle_message(self, message: InMessage):
        print(f"Inbound message: {message=}")

        if isinstance(message, InMessageConnect):
            self._handle_connect(message)
        elif isinstance(message, InMessageDisconnect):
            self._handle_disconnect(message)
        elif isinstance(message, InMessageUpdateSettings):
            self._handle_update_settings(message)
        elif isinstance(message, InMessageUpdateSlos):
            self._handle_update_slos(message)
        elif isinstance(message, InMessageMetrics):
            self._handle_metrics(message)
        else:
            self.message_sender.error(OutMessageError.ERR_WRONG_MESSAGE, f"Unhandled message {message=}")

    
    def _handle_connect(self, message: InMessageConnect):
        if self.session.is_connected:
            self.message_sender.error(f"Session already connected, {message=}")
        else:
            print(f"Connected with {message.stream_settings=}")
            self.session.is_connected = True
            self.session.stream_settings = message.stream_settings
            self.session.slos = message.slos
            self.message_sender.send(OutMessageConnect(message.stream_settings.id))

            self.video_streamer.begin_streaming()

            async def fit():
                try:
                    await self.learning_agent.fit(self.data_iterator)
                except Exception as exc:
                    print(f"Agent fit {exc=}")
            asyncio.create_task(fit())

    
    def _handle_disconnect(self, message: InMessageDisconnect):
        self.video_streamer.end_streaming()

        if self.session.is_connected:
            self.session.is_connected = False
            self.data_iterator.close()
            self.learning_agent.stop()
            self.message_sender.send(OutMessageDisconnect())
        else:
            self.message_sender.error(f"Session has never been connected {message=}")

        asyncio.create_task(self.connection.close())


    def _handle_update_settings(self, message: InMessageUpdateSettings):
        if not self.session.is_connected:
            self.message_sender.error(OutMessageError.ERR_GENERIC, f"Session is not connected yet {message=}")
            return
        
        self.session.stream_settings = message.stream_settings


    def _handle_update_slos(self, message: InMessageUpdateSlos):
        if not self.session.is_connected:
            self.message_sender.error(OutMessageError.ERR_GENERIC, f"Session is not connected yet {message=}")
            return
        
        self.session.slos = message.slos


    def _handle_metrics(self, message: InMessageMetrics):
        df = metrics_to_data_frame(message.metrics_batch, self.session)
        self._save_data(df)
        self.data_iterator.add(df)

    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(
            self._metrics_path,
            mode="a",
            header=not os.path.exists(self._metrics_path),
            index=False,
        )
