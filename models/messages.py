from abc import ABC, abstractmethod
from enum import Enum
import json

from .metrics import Metrics
from .stream_settings import StreamSettings
from .service_level_objectives import ServiceLevelObjectives


# Message Types


class InMessageType(Enum):
    CONNECT = 1
    DISCONNECT = 2
    UPDATE_SETTINGS = 3
    UPDATE_SLOS = 4
    METRICS = 5


class OutMessageType(Enum):
    CONNECT = 1
    DISCONNECT = 2
    ERROR = 3
    FRAME = 4
    SUGGEST_SETTINGS = 5


# Abstract Messages


class InMessage(ABC):
    @property
    @abstractmethod
    def type(self) -> InMessageType:
        pass

    @abstractmethod
    def __init__(self, content: bytes):
        pass


class OutMessage(ABC):
    @property
    @abstractmethod
    def type(self) -> OutMessageType:
        pass

    @abstractmethod
    def encode_content(self) -> bytes:
        pass


# Inbound Messages


class InMessageConnect(InMessage):
    @property
    def type(self):
        return InMessageType.CONNECT
    
    def __init__(self, content):
        data = json.loads(content)
        self.stream_settings = StreamSettings(**data["stream_settings"])
        self.slos = ServiceLevelObjectives(**data["slos"])
        

class InMessageDisconnect(InMessage):
    @property
    def type(self):
        return InMessageType.DISCONNECT
    
    def __init__(self, content):
        pass


class InMessageUpdateSettings(InMessage):
    @property
    def type(self):
        return InMessageType.UPDATE_SETTINGS

    def __init__(self, content):
        data = json.loads(content)
        self.stream_settings = StreamSettings(**data["stream_settings"])


class InMessageUpdateSlos(InMessage):
    @property
    def type(self):
        return InMessageType.UPDATE_SLOS
    
    def __init__(self, content):
        data = json.loads(content)
        self.slos = ServiceLevelObjectives(**data["slos"])


class InMessageMetrics(InMessage):
    @property
    def type(self):
        return InMessageType.METRICS
    
    def __init__(self, content):
        data = json.loads(content)
        self.metrics_batch = [Metrics(**m) for m in data["metrics"]]


# Outbound Messages


class OutMessageConnect(OutMessage):
    def __init__(self, stream_settings_id):
        self.stream_settings_id = stream_settings_id

    @property
    def type(self):
        return OutMessageType.CONNECT
    
    def encode_content(self):
        return json.dumps({
            "stream_settings_id": self.stream_settings_id
        }).encode()


class OutMessageDisconnect(OutMessage):
    @property
    def type(self):
        return OutMessageType.DISCONNECT
    
    def encode_content(self):
        return bytes()
    

class OutMessageError(OutMessage):
    ERR_GENERIC = 1
    ERR_WRONG_MESSAGE = 2

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

    @property
    def type(self):
        return OutMessageType.ERROR
    
    def encode_content(self):
        return json.dumps({
            "code": self.code,
            "message": self.message,
        }).encode()


class OutMessageFrame(OutMessage):
    _divider = b"_"

    def __init__(self, stream: int, frame: bytes, settings: StreamSettings):
        self.stream = stream
        self.frame = frame
        self.settings = settings

    @property
    def type(self):
        return OutMessageType.FRAME
    
    def encode_content(self):
        return (
            self.stream.to_bytes(1) + self._divider
            + self.settings.id.to_bytes(8, signed=True) + self._divider
            + self.frame
        )


class OutMessageSuggestSettings(OutMessage):
    def __init__(self, settings: StreamSettings):
        self.settings = settings

    @property
    def type(self):
        return OutMessageType.SUGGEST_SETTINGS
    
    def encode_content(self):
        return json.dumps({
            "stream_settings": vars(self.settings),
        }).encode()