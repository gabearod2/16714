"""Released student scaffold for HW1: unicycle control under bicycle mismatch."""

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from spark_robot import DiscreteTimeDynamicsConfig

from .unicycle_bicycle_helpers import (
    ModelViewer,
    parse_args,
    rollout,
    save_gain_study,
    save_model_results,
    wrap_angle,
)
from .vehicle_models import (
    DEFAULT_BICYCLE_PARAMS,
    bicycle_parameters,
    bicycle_transition,
    unicycle_transition,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
VELOCITY_GAINS = (1.5, 2.0, 2.5)
HEADING_GAINS = (1.5, 2.5, 3.5)


@dataclass(frozen=True)
class Config:
    dt: float = 0.01
    duration: float = 10.0
    initial_state: tuple = (0.0, 0.0, 0.5, 0.0)  # p_x, p_y, v, theta
    goal: tuple = (5.0, 5.0)
    goal_tolerance: float = 0.2
    velocity_gain: float = 2.0
    heading_gain: float = 2.5
    max_velocity: float = 2.0
    max_acceleration: float = 4.0
    max_turning_rate: float = 1.5
    viewer_fps: float = 50.0
    enable_viewer: bool = False
    show_simulation_info: bool = True
    real_time: bool = False

    @property
    def steps(self):
        return round(self.duration / self.dt)


DEFAULT_CONFIG = Config()


def unicycle_control(state, config):
    """Problem 2.1: implement the controller specified in problem_2.tex."""
    p_x, p_y, velocity, heading = state
    error = np.asarray(config.goal) - np.array([p_x, p_y])
    distance = np.linalg.norm(error)
    if distance <= config.goal_tolerance:
        desired_velocity, heading_error = 0.0, 0.0
        v_dot = config.velocity_gain * (desired_velocity - velocity)
        theta_dot = config.heading_gain * heading_error
    else:
        # TODO(student, Problem 2.1): Implement the errors and [v_dot, theta_dot].
        # errors
        desired_heading = np.arctan2(error[1], error[0])
        heading_error = wrap_angle(desired_heading - heading)
        desired_velocity = min(config.max_velocity, distance) * max(0.0, np.cos(heading_error))
        velocity_error = desired_velocity - velocity

        # control action
        v_dot = config.velocity_gain * velocity_error
        theta_dot = config.heading_gain * heading_error

    return np.array(
        [
            np.clip(v_dot, -config.max_acceleration, config.max_acceleration),
            np.clip(theta_dot, -config.max_turning_rate, config.max_turning_rate),
        ]
    )


def bicycle_initial_state(unicycle_state):
    """Problem 2.2: map [p_x,p_y,v,theta] into the bicycle state."""
    # TODO(student, Problem 2.2): Return [X,Y,v_x,v_y,r,psi].
    p_x, p_y, velocity, heading = unicycle_state
    return np.array([p_x, p_y, velocity, 0, 0, heading])


def bicycle_control(state, config):
    """Problem 2.2: implement the provided pure-rolling command conversion."""
    # TODO(student, Problem 2.2): Implement the conversion given in
    # problem_2.tex.
    X,Y,v_x,v_y,r,psi = state
    unicycle_state = np.array([X, Y, v_x, psi])
    v_dot, theta_dot = unicycle_control(unicycle_state, config)
    a_x = v_dot
    L = DEFAULT_BICYCLE_PARAMS.front_length + DEFAULT_BICYCLE_PARAMS.rear_length
    delta = np.arctan(L * theta_dot / v_x)
    return np.array(
        [
            a_x,
            np.clip(
                delta, 
                -DEFAULT_BICYCLE_PARAMS.steering_limit, 
                DEFAULT_BICYCLE_PARAMS.steering_limit
            )
        ]
    )


class ComparisonPipeline:
    def __init__(self, config=DEFAULT_CONFIG, output_dir=OUTPUT_DIR):
        self.config, self.output_dir = config, Path(output_dir)
        self.unicycle_model, self.bicycle_model = self.make_models()

    def make_models(self):
        unicycle = DiscreteTimeDynamicsConfig(
            4, 2, unicycle_transition, dynamics_variant="unicycle_4_state",
            parameters={"dt": self.config.dt},
            state_names=("p_x", "p_y", "v", "theta"),
            control_names=("v_dot", "theta_dot"),
        )
        bicycle = DiscreteTimeDynamicsConfig(
            6, 2, bicycle_transition, dynamics_variant="bicycle_dynamic_6_state",
            parameters=bicycle_parameters(self.config.dt),
            state_names=("X", "Y", "v_x", "v_y", "r", "psi"),
            control_names=("a_x", "delta"),
        )
        return unicycle.create_dynamics_model(), bicycle.create_dynamics_model()

    def simulate(self, model_name):
        if model_name == "unicycle":
            model, initial = self.unicycle_model, self.config.initial_state
            policy = lambda state: unicycle_control(state, self.config)
        elif model_name == "bicycle":
            model, initial = self.bicycle_model, bicycle_initial_state(self.config.initial_state)
            policy = lambda state: bicycle_control(state, self.config)
        else:
            raise ValueError('model_name must be "unicycle" or "bicycle".')
        states, controls = rollout(model, policy, initial, self.config.steps, self.config.dt)
        times = np.arange(self.config.steps + 1) * self.config.dt
        return model, times, states, controls

    def run(self, model_name):
        model, times, states, controls = self.simulate(model_name)
        model_dir = save_model_results(
            model_name, times, states, controls, model, self.config.goal, self.output_dir
        )
        if self.config.enable_viewer:
            viewer = ModelViewer(self.config)
            try:
                viewer.replay(states, model)
            finally:
                viewer.close()
        goal_error = float(np.linalg.norm(states[-1, :2] - self.config.goal))
        print(f"Saved {model_name} results to {model_dir.resolve()}")
        print(f"Final goal error: {goal_error:.3f} m")
        return {"times": times, "states": states, "controls": controls,
                "goal_error": goal_error, "output_dir": model_dir}


def run_gain_study(config=DEFAULT_CONFIG, output_dir=OUTPUT_DIR):
    """Provided study for Problem 2.3; students analyze the two plots."""
    traces = []
    for velocity_gain, heading_gain in product(VELOCITY_GAINS, HEADING_GAINS):
        trial = replace(config, velocity_gain=velocity_gain,
                        heading_gain=heading_gain, enable_viewer=False)
        pipeline = ComparisonPipeline(trial, output_dir)
        _, times, unicycle_x, _ = pipeline.simulate("unicycle")
        _, _, bicycle_x, _ = pipeline.simulate("bicycle")
        position_error = np.linalg.norm(unicycle_x[:, :2] - bicycle_x[:, :2], axis=1)
        orientation_error = np.abs(wrap_angle(unicycle_x[:, 3] - bicycle_x[:, 5]))
        traces.append((velocity_gain, heading_gain, position_error, orientation_error))
    study_dir = save_gain_study(times, traces, output_dir)
    print(f"Saved gain study to {study_dir.resolve()}")
    return {"times": times, "traces": traces, "output_dir": study_dir}


def run(model_name, config=DEFAULT_CONFIG, output_dir=OUTPUT_DIR, **overrides):
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    return ComparisonPipeline(config, output_dir).run(model_name)


def main(argv=None):
    args = parse_args(argv, OUTPUT_DIR)
    if args.gain_study:
        return run_gain_study(output_dir=args.output_dir)
    return run(args.model, output_dir=args.output_dir, enable_viewer=args.viewer,
               real_time=args.real_time, show_simulation_info=args.show_simulation_info)


if __name__ == "__main__":
    main()
