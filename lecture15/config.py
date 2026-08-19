from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture15Config:
    seed: int = 714
    horizon: int = 101
    dt: float = 1.0
    process_variance: float = 0.01
    measurement_variance: float = 0.01


DEFAULT_CONFIG = Lecture15Config()

