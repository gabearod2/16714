from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture2Config:
    single_dt: float = 0.1
    single_duration: float = 5.0
    double_dt: float = 1.0
    double_duration: float = 5.0
    vehicle_dt: float = 0.1
    vehicle_steps: int = 50
    bicycle_dt: float = 0.01
    bicycle_steps: int = 500
    bicycle_cross_track_gain: float = 2.0
    bicycle_velocity_gain: float = 1.2
    bicycle_goal_speed_gain: float = 0.8
    bicycle_max_reference_speed: float = 8.0
    goal: tuple[float, float] = (5.0, 5.0)
    enable_viewer: bool = False
    viewer_show_simulation_info: bool = True
    real_time: bool = False


DEFAULT_CONFIG = Lecture2Config()
