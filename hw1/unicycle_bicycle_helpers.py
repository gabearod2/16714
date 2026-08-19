"""Saving and viewer support for the independent HW1 model runs."""

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_agent import AgiBotG1MobileBaseAgent
from spark_robot import AgiBotG1MobileBaseDynamic1Config
from spark_utils import VizColor


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rollout(model, policy, initial_state, steps, dt):
    states = np.empty((steps + 1, model.state_dim))
    controls = np.empty((steps, model.control_dim))
    states[0] = initial_state
    for step in range(steps):
        controls[step] = policy(states[step])
        states[step + 1] = model.step(states[step], controls[step], dt, "native")
    return states, controls


def parse_args(argv, output_dir):
    parser = argparse.ArgumentParser(description="Run one HW1 vehicle model")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=("unicycle", "bicycle"))
    mode.add_argument("--gain-study", action="store_true")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument(
        "--show-simulation-info", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    return parser.parse_args(argv)


def _save_curve(path, x, y, *, xlabel, ylabel, title, goal=None, equal=False):
    fig, axis = plt.subplots(figsize=(5.6, 4.4))
    axis.plot(x, y, color="tab:blue")
    if goal is not None:
        axis.scatter(*goal, color="tab:red", marker="*", s=100)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(True)
    if equal:
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_results(model_name, times, states, controls, model, goal, output_dir):
    """Save CSV traces and one curve per figure for one model."""
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(model_dir / "states.csv", np.column_stack([times, states]), delimiter=",",
               header=",".join(("time",) + model.state_names), comments="")
    np.savetxt(model_dir / "controls.csv", np.column_stack([times[:-1], controls]), delimiter=",",
               header=",".join(("time",) + model.control_names), comments="")
    _save_curve(model_dir / "xy_trajectory.png", states[:, 0], states[:, 1],
                xlabel="x [m]", ylabel="y [m]", title=f"{model_name.title()} XY trajectory",
                goal=np.asarray(goal), equal=True)
    for index, state_name in enumerate(model.state_names):
        values, ylabel = states[:, index], state_name
        if state_name in ("theta", "psi"):
            values, ylabel = np.unwrap(values), f"{state_name} (unwrapped)"
        _save_curve(model_dir / f"state_{state_name}.png", times, values,
                    xlabel="time [s]", ylabel=ylabel,
                    title=f"{model_name.title()}: {state_name} vs. time")
    return model_dir


def save_gain_study(times, traces, output_dir):
    """Save position/orientation model-error histories for every gain pair."""
    study_dir = Path(output_dir) / "gain_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for velocity_gain, heading_gain, position_error, orientation_error in traces:
        rows.append(np.column_stack([
            times, np.full_like(times, velocity_gain), np.full_like(times, heading_gain),
            position_error, orientation_error,
        ]))
    np.savetxt(study_dir / "errors.csv", np.vstack(rows), delimiter=",",
               header="time,velocity_gain,heading_gain,position_error,orientation_error",
               comments="")
    for index, filename, ylabel, title in (
        (2, "position_error.png", "position error [m]", "Position model error"),
        (3, "orientation_error.png", "orientation error [rad]", "Orientation model error"),
    ):
        fig, axis = plt.subplots(figsize=(6.4, 4.8))
        for velocity_gain, heading_gain, *errors in traces:
            axis.plot(times, errors[index - 2], label=fr"$k_v={velocity_gain:g}, k_\theta={heading_gain:g}$")
        axis.set(xlabel="time [s]", ylabel=ylabel, title=title)
        axis.grid(True)
        axis.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(study_dir / filename, dpi=180)
        plt.close(fig)
    return study_dir


class ModelViewer:
    """Replay one externally generated numerical trajectory."""

    def __init__(self, config):
        self.config = config
        self.robot_config = AgiBotG1MobileBaseDynamic1Config()
        self.default_q = np.array(
            [self.robot_config.DefaultDoFVal[dof] for dof in self.robot_config.DoFs]
        )
        self.agent = AgiBotG1MobileBaseAgent(
            self.robot_config,
            enable_viewer=True,
            viewer_show_simulation_info=config.show_simulation_info,
            enable_keyboard_control=False,
            enable_camera=False,
            use_sim_dynamics=False,
            dt=config.dt,
            control_decimation=1,
            real_time=False,
            obstacle_debug={"num_obstacle": 0},
        )

    @staticmethod
    def _path(states, height):
        return np.column_stack([states[:, :2], np.full(len(states), height)])

    def replay(self, states, model):
        states = np.asarray(states)
        display_stride = max(1, int(np.ceil(1.0 / (self.config.viewer_fps * self.config.dt))))
        display_indices = self._sample_indices(len(states), display_stride)
        path_stride = max(display_stride, int(np.ceil(len(states) / 200)))
        path_indices = self._sample_indices(len(states), path_stride)
        path = self._path(states[path_indices], 0.045)
        heading_name = "theta" if "theta" in model.state_names else "psi"
        heading_index = model.state_names.index(heading_name)
        goal = np.asarray(self.config.goal)
        self.agent.set_viewer_simulation_info(
            {
                "Experiment": f"HW1 {model.variant}",
                "Robot config": type(model.robot_cfg).__name__,
                "Dynamics model": model.variant,
                "Dynamics order": "native discrete",
                "Dimensions": f"state={model.state_dim}, control={model.control_dim}",
                "Propagation": "external forward Euler replay",
                "Control period": f"{self.config.dt:g} s",
                "State order": ", ".join(model.state_names),
                "Playback": f"{self.config.viewer_fps:g} FPS",
                "Simulation time": "0.000 s",
            },
            replace=True,
        )

        # The viewer replays the numerical rollout; it does not propagate dynamics.
        playback_start = time.perf_counter()
        for frame, source_step in enumerate(display_indices):
            if not self.agent.is_running():
                break
            if self.config.real_time and frame:
                deadline = playback_start + source_step * self.config.dt
                time.sleep(max(0.0, deadline - time.perf_counter()))
            state = states[source_step]
            self.agent.set_viewer_simulation_info(
                {"Simulation time": f"{source_step * self.config.dt:.3f} s"}
            )
            q = self.default_q.copy()
            q[self.robot_config.DoFs.LinearX] = state[0]
            q[self.robot_config.DoFs.LinearY] = state[1]
            q[self.robot_config.DoFs.RotYaw] = state[heading_index]
            self.agent.reset({"reset_dof_pos": q})
            self.agent.begin_render_frame()
            path_count = np.searchsorted(path_indices, source_step, side="right")
            self._draw(path[:path_count], (0.12, 0.42, 1.0, 0.9))
            self.agent.render_sphere(
                np.array([goal[0], goal[1], 0.09]),
                np.eye(3),
                np.full(3, 0.12),
                VizColor.goal,
            )
            self.agent.render()

    @staticmethod
    def _sample_indices(length, stride):
        indices = np.arange(0, length, stride, dtype=int)
        if indices[-1] != length - 1:
            indices = np.append(indices, length - 1)
        return indices

    def _draw(self, path, color):
        for start, end in zip(path[:-1], path[1:]):
            self.agent.render_line_segment(start, end, 0.014, color)

    def close(self):
        self.agent.close_viewer()
