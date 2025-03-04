import asyncio
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

from models.messages import *
from utils.message_coder import MessageCoder


class MessageSender:
    def __init__(
        self,
        connection: ServerConnection,
        message_coder: MessageCoder,
    ):
        self.connection = connection
        self.message_coder = message_coder


    def send(self, message: OutMessage):
        encoded_message = self.message_coder.encode_out_message(message)

        asyncio.create_task(self._send(encoded_message))


    def error(self, code: int, message: str):
        print(f"ERROR: {message}")
        self.send(OutMessageError(code, message))


    async def _send(self, encoded_message: bytes):
        if self.connection.state != State.OPEN:
            return
        try:
            await self.connection.send(encoded_message)
        except:
            pass
