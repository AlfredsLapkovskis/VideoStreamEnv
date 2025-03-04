import asyncio
import time

from models.messages import *
from services.message_sender import MessageSender


class MessageSenderWithRateLimit(MessageSender):
    def __init__(self, connection, message_coder, rate_limit=None):
        super().__init__(connection, message_coder)

        self.rate_limit = rate_limit
        if self.rate_limit is not None:
            self.last_time = time.monotonic()
            self.lock = asyncio.Lock()
            self.send_tokens = rate_limit
            self._settings_id = None
            self._tasks = []

    
    def send(self, message):
        if isinstance(message, OutMessageFrame) and self.rate_limit is not None:
            if self._settings_id != message.settings.id:
                self._settings_id = message.settings.id
                for task in self._tasks:
                    task.cancel()
                self._tasks.clear()
            encoded_message = self.message_coder.encode_out_message(message)
            self._tasks.append(asyncio.create_task(self._send_with_rate_limit(encoded_message)))
        else:
            super().send(message)


    async def _send_with_rate_limit(self, encoded_message):
        async with self.lock:
            data_len = len(encoded_message)
            current_time = time.monotonic()
            elapsed = current_time - self.last_time

            self.send_tokens += elapsed * self.rate_limit
            if self.send_tokens > self.rate_limit:
                self.send_tokens = self.rate_limit
            
            if data_len > self.send_tokens:
                deficit = data_len - self.send_tokens
                wait_time = deficit / self.rate_limit
                try:
                    await asyncio.sleep(wait_time)
                except BaseException as exc:
                    print(f"Send with rate limit {exc=}")
                    return
                finally:
                    self.send_tokens = 0
                    self.last_time = current_time + wait_time
            else:
                self.send_tokens -= data_len
                self.last_time = current_time

        await super()._send(encoded_message)
