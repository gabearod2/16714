"""Configuration for Lecture 19 model-reference adaptive control."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Lecture19Config:
    seed: int = 714
    state_dim: int = 2
    control_dim: int = 2
    horizon: int = 29
    dt: float = 1.0
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -1.0])
    )


DEFAULT_CONFIG = Lecture19Config()

