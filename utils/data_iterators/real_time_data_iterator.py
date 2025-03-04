import pandas as pd
import asyncio
from asyncio import CancelledError

from models.messages import OutMessageSuggestSettings
from services.message_sender import MessageSender
from utils.data_iterators.data_iterator import DataIterator


class RealTimeDataIterator(DataIterator):

    def __init__(self, message_sender: MessageSender):
        self._df = pd.DataFrame()
        self._future_df: asyncio.Future = None
        self.message_sender = message_sender
        self._is_closed = False


    async def request_next(self, settings):
        if self._is_closed:
            raise CancelledError()
        if settings:
            self.message_sender.send(OutMessageSuggestSettings(settings))

        assert self._future_df is None

        self._future_df = asyncio.Future()
        try:
            df = await self._future_df
        finally:
            self._future_df = None

        return df


    def add(self, df: pd.DataFrame):
        if self._is_closed:
            raise RuntimeError("RealTimeAsyncIterator has already stopped.")
        if self._future_df:
            self._future_df.set_result(pd.concat([self._df, df]))
            self._df = pd.DataFrame()
        else:
            self._df = pd.concat([self._df, df])


    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        if self._future_df:
            self._future_df.set_exception(CancelledError())
            self._future_df = None