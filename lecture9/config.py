from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture9Config:
    initial_state: tuple[float, ...] = (10.0, 0.0)
    goal_state: tuple[float, ...] = (0.0, 0.0)
    dt: float = 0.5
    horizon_steps: int = 10
    integrator: str = "ZOH"
    default_case: str = "linear_mpc"
    enable_viewer: bool = False
    viewer_show_simulation_info: bool = True


DEFAULT_CONFIG = Lecture9Config()
