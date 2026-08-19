from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture16Config:
    seed: int = 714
    horizon: int = 10
    dt: float = 1.0
    process_variance: float = 0.001
    measurement_variance: float = 0.001


DEFAULT_CONFIG = Lecture16Config()

