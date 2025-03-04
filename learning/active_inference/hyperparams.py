from dataclasses import dataclass, field

from learning.base_hyperparams import BaseHyperParams


@dataclass
class HyperParams(BaseHyperParams):
    surprise_buffer_size: int = 10
    surprise_threshold_factor: float = 2.0
    weight_of_past_data: float = 0.6
    initial_additional_surprises: list[list[float | int]] = field(default_factory=lambda: [
        [0, 0, 0, 1.],
        [0, 0, -1, 1.],
        [0, -1, 0, 1.],
        [0, -1, -1, 1.],
        [-1, 0, 0, 1.],
        [-1, 0, -1, 1.],
        [-1, -1, 0, 1.],
        [-1, -1, -1, 1.],
    ])
    graph_max_indegree: int = 8
    hill_climb_epsilon: float = 1.0
