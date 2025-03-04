from dataclasses import dataclass, field

from learning.base_hyperparams import BaseHyperParams


@dataclass
class DQNHyperParams(BaseHyperParams):
    batch_size: int = 128
    learning_rate: float = 1e-4
    initial_eps: float = 1.
    final_eps: float = 0.05
    exploration_fraction: float = 0.1
    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    discount_factor: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 4
    target_update_interval: int = 10_000


@dataclass
class A2CHyperParams(BaseHyperParams):
    batch_size: int = 64
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.9
    vf_coef: float = 0.75
    ent_coef: float = 0.01
    normalize_advantage: bool = True
    net_arch: dict[str: list[int]] = field(default_factory=lambda: dict(pi=[128, 128], vf=[128, 128]))


@dataclass
class PPOHyperParams(BaseHyperParams):
    batch_size: int = 128
    learning_rate: float = 1e-4
    n_steps: int = 1280
    n_epochs: int = 10
    discount_factor: int = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = False
    vf_coef: float = 0.25
    ent_coef: float = 0.01
    net_arch: dict[str: list[int]] = field(default_factory=lambda: dict(pi=[64, 64], vf=[64, 64]))
