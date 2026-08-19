from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture3Config:
    seed: int = 714
    dt: float = 0.1
    steps: int = 50
    random_velocity_std: float = 0.35
    joint_gain: float = 0.7
    cartesian_gain: float = 1.0
    jacobian_damping: float = 0.05
    goal_joint_offset: tuple[float, ...] = (
        0.25,
        0.15,
        -0.20,
        0.25,
        0.15,
        -0.15,
        0.20,
    )
    enable_viewer: bool = False
    viewer_show_simulation_info: bool = True
    real_time: bool = False


DEFAULT_CONFIG = Lecture3Config()
