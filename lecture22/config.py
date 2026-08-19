"""Configuration for Lecture 22 policy-gradient and ILC comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture22Config:
    seed: int = 714
    scenario: str = "non-repetitive"
    A: float = 0.5
    B: float = 1.0
    Q: float = 1.0
    R: float = 1.0
    initial_state: float = 0.1
    horizon: int = 50
    dt: float = 1.0
    episodes: int = 20
    actor_step: float = 1.0e-4
    critic_step: float = 1.0e-2
    initial_mu: float = -0.1
    initial_sigma: float = 0.1
    initial_value_weight: float = 3.0
    tolerance: float = -1.0
    mu_bounds: tuple[float, float] = (-5.0, 5.0)
    sigma_bounds: tuple[float, float] = (1.0e-4, 5.0)


DEFAULT_CONFIG = Lecture22Config()

