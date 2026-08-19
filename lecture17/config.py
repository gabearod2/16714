from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture17Config:
    horizon: int = 100
    dt: float = 1.0
    initial_state: float = 1.0
    nominal_parameter: float = 1.0
    variation: float = 0.05
    rls_information: float = 0.5
    forgetting_factor: float = 0.5


DEFAULT_CONFIG = Lecture17Config()

