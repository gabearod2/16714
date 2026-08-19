"""Configuration for Lecture 21 value-learning experiments."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Lecture21Config:
    seed: int = 714
    A: float = 1.0
    B: float = 0.5
    Q: float = 1.0
    R: float = 1.0
    tolerance: float = 0.01
    horizon: int = 100
    dt: float = 1.0
    initial_state: float = 1.0
    episodes: int = 100
    learning_rate: float = 1.0
    exploration_rate: float = 0.1
    initial_q_weights: np.ndarray = field(
        default_factory=lambda: np.array([[4.0, 1.0], [1.0, 4.0]])
    )


DEFAULT_CONFIG = Lecture21Config()

