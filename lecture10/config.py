from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture10MPCConfig:
    initial_state: tuple[float, float] = (10.0, 0.0)
    goal_state: tuple[float, float] = (0.0, 0.0)
    dt: float = 0.5
    horizon_steps: int = 10
    execution_steps: int = 30
    integrator: str = "ZOH"
    default_case: str = "constrained_mpc"
    enable_viewer: bool = False
    viewer_show_simulation_info: bool = True


DEFAULT_CONFIG = Lecture10MPCConfig()
