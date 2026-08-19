from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture18Config:
    seed: int = 714
    dt: float = 0.2
    horizon: int = 50
    process_variance: float = 0.1
    measurement_variance: float = 0.01


DEFAULT_CONFIG = Lecture18Config()

