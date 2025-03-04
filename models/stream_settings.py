from dataclasses import dataclass


@dataclass
class StreamSettings:
    id: int
    n_streams: int
    fps: int
    resolution: int
