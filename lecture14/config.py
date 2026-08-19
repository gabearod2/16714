from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture14Config:
    seed: int = 714
    samples: int = 100


DEFAULT_CONFIG = Lecture14Config()

