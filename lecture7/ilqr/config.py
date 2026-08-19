from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture7ILQRConfig:
    base_goal: tuple[float, float, float] = (0.5, 0.25, 0.793)
    goal_state: tuple[float, float, float] = (0.5, 0.25, 0.3)
    horizon_steps: int = 150
    dt: float = 0.1
    integrator: str = "Euler"
    max_iterations: int = 120
    execution_steps: int = 150
    enable_viewer: bool = True
    viewer_show_simulation_info: bool = True


DEFAULT_CONFIG = Lecture7ILQRConfig()
