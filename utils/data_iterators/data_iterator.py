from abc import ABC, abstractmethod

from models.messages import ABC, StreamSettings
from models.stream_settings import StreamSettings


class DataIterator(ABC):
    @abstractmethod
    async def request_next(self, settings: StreamSettings | None):
        pass
