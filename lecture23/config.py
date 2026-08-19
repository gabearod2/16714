"""Configuration for Lecture 23 policy-gradient experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture23Config:
    seed: int = 714
    A: float = 1.0
    B: float = 0.5
    Q: float = 1.0
    R: float = 1.0
    initial_state: float = 1.0
    tolerance: float = 0.01
    horizon: int = 9
    dt: float = 1.0
    episodes: int = 200
    actor_step: float = 0.01
    critic_step: float = 1.0
    initial_mu: float = -0.1
    initial_sigma: float = 0.1
    initial_value_weight: float = 3.0
    mu_bounds: tuple[float, float] = (-5.0, 5.0)
    sigma_bounds: tuple[float, float] = (1.0e-4, 5.0)


DEFAULT_CONFIG = Lecture23Config()

