from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture7LQRConfig:
    initial_state: tuple[float, ...] = (10.0, 10.0, 1.0, 5.0)
    goal_state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    dt: float = 0.5
    horizon_steps: int = 20
    agent_dt: float = 0.01
    integrator: str = "ZOH"
    enable_viewer: bool = True
    viewer_show_simulation_info: bool = True


DEFAULT_CONFIG = Lecture7LQRConfig()
