"""Configuration for Lecture 12 frequency-domain ILC."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture12Config:
    horizon: int = 100
    iterations: int = 10
    dt: float = 1.0
    disturbance_sigma: float = 0.0
    seed: int = 714
    kp: float = 0.1
    kd: float = 0.5
    integrator: str = "Euler"


DEFAULT_CONFIG = Lecture12Config()

