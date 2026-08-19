from dataclasses import asdict, replace
import time

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from spark_agent import AgiBotG1MobileBaseAgent
from spark_robot import (
    AgiBotG1MobileBaseBicycleDynamic2Config,
    AgiBotG1MobileBaseDynamic1Config,
    AgiBotG1MobileBaseDynamic2Config,
    AgiBotG1MobileBaseUnicycleDynamic1Config,
    BicycleParams,
)
from spark_utils import VizColor
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from .config import DEFAULT_CONFIG, Lecture2Config


DATA_OUTPUT_DIR = data_dir("lecture2")


class Lecture2VehicleViewer:
    """Replay Lecture 2 planar states with the SPARK AgiBot model."""

    def __init__(self, config):
        self.config = config
        self.robot_cfg = AgiBotG1MobileBaseDynamic1Config()
        self.default_q = np.array(
            [self.robot_cfg.DefaultDoFVal[dof] for dof in self.robot_cfg.DoFs],
            dtype=float,
        )
        self.agent = AgiBotG1MobileBaseAgent(
            self.robot_cfg,
            enable_viewer=True,
            viewer_show_simulation_info=config.viewer_show_simulation_info,
            enable_keyboard_control=False,
            enable_camera=False,
            use_sim_dynamics=False,
            dt=min(config.vehicle_dt, config.bicycle_dt),
            control_decimation=1,
            real_time=False,
            obstacle_debug={"num_obstacle": 0},
            viewer_config={
                "camera_lookat": (2.5, 2.5, 0.6),
                "camera_distance": 10.5,
                "camera_azimuth": 135.0,
                "camera_elevation": -35.0,
            },
        )
        # Directional-light shadows are clipped to a finite box around the
        # model origin.  Lecture 2 drives several metres beyond that box, so
        # disable them to keep the robot and path appearance spatially uniform.
        self.agent.model.light_castshadow[:] = 0
        floor_geom_id = mujoco.mj_name2id(
            self.agent.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        if floor_geom_id != -1:
            floor_material_id = int(self.agent.model.geom_matid[floor_geom_id])
            if floor_material_id != -1:
                self.agent.model.mat_reflectance[floor_material_id] = 0.0

    def replay(
        self,
        planar_states,
        *,
        sample_dt,
        goal,
        dynamics_model,
        integrator,
        experiment,
    ):
        planar_states = np.asarray(planar_states, dtype=float).reshape(-1, 3)
        goal = np.asarray(goal, dtype=float).reshape(2)
        self.agent.set_viewer_simulation_info(
            {
                "Experiment": experiment,
                "Robot config": type(dynamics_model.robot_cfg).__name__,
                "Dynamics model": dynamics_model.variant,
                "Dynamics order": dynamics_model.order,
                "Dimensions": (
                    f"state={dynamics_model.state_dim}, "
                    f"control={dynamics_model.control_dim}"
                ),
                "Propagation": f"trajectory replay ({integrator})",
                "Control period": f"{float(sample_dt):g} s",
            },
            replace=True,
        )
        path = np.column_stack(
            [planar_states[:, :2], 0.025 * np.ones(len(planar_states))]
        )
        for step, state in enumerate(planar_states):
            if not self.agent.is_running():
                return
            self.agent.set_viewer_simulation_info(
                {"Simulation time": f"{step * float(sample_dt):.3f} s"}
            )
            q = self.default_q.copy()
            q[self.robot_cfg.DoFs.LinearX] = state[0]
            q[self.robot_cfg.DoFs.LinearY] = state[1]
            q[self.robot_cfg.DoFs.RotYaw] = state[2]
            self.agent.reset({"reset_dof_pos": q})
            self.agent.begin_render_frame()
            for start, end in zip(path[:-1], path[1:]):
                self.agent.render_line_segment(
                    start,
                    end,
                    0.012,
                    (0.15, 0.45, 1.0, 0.75),
                )
            self.agent.render_sphere(
                np.array([goal[0], goal[1], 0.08]),
                np.eye(3),
                0.12 * np.ones(3),
                VizColor.goal,
            )
            self.agent.render()
            if self.config.real_time:
                time.sleep(float(sample_dt))

    def close(self):
        self.agent.close_viewer()


def run(
    save_dir=DATA_OUTPUT_DIR,
    config: Lecture2Config = DEFAULT_CONFIG,
    *,
    enable_viewer=None,
    viewer_show_simulation_info=None,
    real_time=None,
):
    if enable_viewer is not None:
        config = replace(config, enable_viewer=bool(enable_viewer))
    if viewer_show_simulation_info is not None:
        config = replace(
            config,
            viewer_show_simulation_info=bool(viewer_show_simulation_info),
        )
    if real_time is not None:
        config = replace(config, real_time=bool(real_time))

    save_dir = ensure_dir(save_dir)
    viewer = Lecture2VehicleViewer(config) if config.enable_viewer else None
    try:
        summaries = {}
        summaries["single_integrator"] = _single_integrator_comparison(
            save_dir, config, viewer
        )
        summaries["double_integrator"] = _double_integrator_comparison(
            save_dir, config, viewer
        )
        summaries["unicycle3"] = _unicycle_example(save_dir, config, viewer)
        summaries["bicycle_dynamic"] = _bicycle_example(save_dir, config, viewer)
    finally:
        if viewer is not None:
            viewer.close()
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summaries)
    return summaries


def _single_integrator_comparison(save_dir, config=DEFAULT_CONFIG, viewer=None):
    model = AgiBotG1MobileBaseDynamic1Config().create_dynamics_model(
        state_dof_names=["LinearX"], control_names=["vLinearX"]
    )
    x0 = np.array([0.0])
    tmax = config.single_duration
    dt = config.single_dt
    steps = round(tmax / dt)

    def controller(x, t, _k=None):
        return np.array([5.0 - x[0]], dtype=float)

    t_ct, x_ct, u_ct = _continuous_reference(model, controller, x0, tmax, dt)
    args = (model.robot_cfg, controller, x0, steps, dt)
    selection = dict(state_dof_names=["LinearX"], control_names=["vLinearX"])
    t_zoh, x_zoh, u_zoh = _rollout(*args, integrator="ZOH", **selection)
    t_euler, x_euler, u_euler = _rollout(*args, integrator="Euler", **selection)
    t_rk4, x_rk4, u_rk4 = _rollout(*args, integrator="RK4", **selection)

    write_csv(
        save_dir / "single_integrator_states.csv",
        ("t", "ct", "zoh", "euler", "rk4"),
        np.column_stack([t_zoh, x_ct[:, 0], x_zoh[:, 0], x_euler[:, 0], x_rk4[:, 0]]),
    )
    write_csv(
        save_dir / "single_integrator_controls.csv",
        ("t", "ct", "zoh", "euler", "rk4"),
        np.column_stack([t_zoh[:-1], u_ct[:, 0], u_zoh[:, 0], u_euler[:, 0], u_rk4[:, 0]]),
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=False)
    axes[0].plot(t_ct, x_ct[:, 0], "k", label="continuous")
    axes[0].plot(t_zoh, x_zoh[:, 0], "r", label="zoh")
    axes[0].plot(t_euler, x_euler[:, 0], "b", label="euler")
    axes[0].plot(t_rk4, x_rk4[:, 0], "g", label="rk4")
    axes[0].set_title("Single Integrator State")
    axes[1].plot(t_ct[:-1], u_ct[:, 0], "k", label="continuous")
    axes[1].plot(t_zoh[:-1], u_zoh[:, 0], "r", label="zoh")
    axes[1].plot(t_euler[:-1], u_euler[:, 0], "b", label="euler")
    axes[1].plot(t_rk4[:-1], u_rk4[:, 0], "g", label="rk4")
    axes[1].set_title("Single Integrator Control")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "single_integrator_plot.png", dpi=160)
    plt.close(fig)
    if viewer is not None:
        viewer.replay(
            np.column_stack([x_rk4[:, 0], np.zeros((len(x_rk4), 2))]),
            sample_dt=dt,
            goal=(5.0, 0.0),
            dynamics_model=model,
            integrator="RK4",
            experiment="single-integrator comparison",
        )

    return {
        "final_ct": x_ct[-1, 0],
        "final_zoh": x_zoh[-1, 0],
        "final_euler": x_euler[-1, 0],
        "final_rk4": x_rk4[-1, 0],
    }


def _double_integrator_comparison(save_dir, config=DEFAULT_CONFIG, viewer=None):
    model = AgiBotG1MobileBaseDynamic2Config().create_dynamics_model(
        state_dof_names=["LinearX"], control_names=["aLinearX"]
    )
    x0 = np.array([0.0, 0.0])
    tmax = config.double_duration
    dt = config.double_dt
    steps = round(tmax / dt)

    def controller(x, t, _k=None):
        return np.array([5.0 - x[0] - 2.0 * x[1]], dtype=float)

    t_ct, x_ct, u_ct = _continuous_reference(model, controller, x0, tmax, dt)
    args = (model.robot_cfg, controller, x0, steps, dt)
    selection = dict(state_dof_names=["LinearX"], control_names=["aLinearX"])
    t_zoh, x_zoh, u_zoh = _rollout(*args, integrator="ZOH", **selection)
    t_euler, x_euler, u_euler = _rollout(*args, integrator="Euler", **selection)
    t_rk4, x_rk4, u_rk4 = _rollout(*args, integrator="RK4", **selection)

    write_csv(
        save_dir / "double_integrator_states.csv",
        ("t", "ct_x", "ct_v", "zoh_x", "zoh_v", "euler_x", "euler_v", "rk4_x", "rk4_v"),
        np.column_stack([t_zoh, x_ct, x_zoh, x_euler, x_rk4]),
    )
    write_csv(
        save_dir / "double_integrator_controls.csv",
        ("t", "ct", "zoh", "euler", "rk4"),
        np.column_stack([t_zoh[:-1], u_ct[:, 0], u_zoh[:, 0], u_euler[:, 0], u_rk4[:, 0]]),
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=False)
    labels = (("x", 0), ("v", 1))
    for ax, (name, idx) in zip(axes[:2], labels):
        ax.plot(t_ct, x_ct[:, idx], "k", label="continuous")
        ax.plot(t_zoh, x_zoh[:, idx], "r", label="zoh")
        ax.plot(t_euler, x_euler[:, idx], "b", label="euler")
        ax.plot(t_rk4, x_rk4[:, idx], "g", label="rk4")
        ax.set_title(f"Double Integrator {name}")
    axes[2].plot(t_ct[:-1], u_ct[:, 0], "k", label="continuous")
    axes[2].plot(t_zoh[:-1], u_zoh[:, 0], "r", label="zoh")
    axes[2].plot(t_euler[:-1], u_euler[:, 0], "b", label="euler")
    axes[2].plot(t_rk4[:-1], u_rk4[:, 0], "g", label="rk4")
    axes[2].set_title("Double Integrator Control")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "double_integrator_plot.png", dpi=160)
    plt.close(fig)
    if viewer is not None:
        viewer.replay(
            np.column_stack([x_rk4[:, 0], np.zeros((len(x_rk4), 2))]),
            sample_dt=dt,
            goal=(5.0, 0.0),
            dynamics_model=model,
            integrator="RK4",
            experiment="double-integrator comparison",
        )

    return {
        "final_ct": x_ct[-1],
        "final_zoh": x_zoh[-1],
        "final_euler": x_euler[-1],
        "final_rk4": x_rk4[-1],
    }


def _unicycle_example(save_dir, config=DEFAULT_CONFIG, viewer=None):
    model = AgiBotG1MobileBaseUnicycleDynamic1Config().create_dynamics_model()
    goal = np.asarray(config.goal, dtype=float)
    x0 = np.zeros(3)
    dt = config.vehicle_dt
    steps = config.vehicle_steps

    def controller(x, _t, _k):
        heading = np.array([np.cos(x[2]), np.sin(x[2])])
        v = np.dot(goal - x[:2], heading)
        omega = np.arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2]
        return np.array([v, omega], dtype=float)

    t, states, controls = _rollout(
        model.robot_cfg, controller, x0, steps, dt, integrator="Euler"
    )
    write_csv(save_dir / "unicycle3_states.csv", ("t", "x", "y", "theta"), np.column_stack([t, states]))
    write_csv(save_dir / "unicycle3_controls.csv", ("t", "v", "omega"), np.column_stack([t[:-1], controls]))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(states[:, 0], states[:, 1], "k", label="path")
    ax.plot(goal[0], goal[1], "r*", markersize=8, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Unicycle3 Path")
    fig.tight_layout()
    fig.savefig(save_dir / "unicycle3_path.png", dpi=160)
    plt.close(fig)
    if viewer is not None:
        viewer.replay(
            states,
            sample_dt=dt,
            goal=goal,
            dynamics_model=model,
            integrator="Euler",
            experiment="unicycle",
        )
    return {"final_state": states[-1], "goal_error": np.linalg.norm(states[-1, :2] - goal)}


def _bicycle_example(save_dir, config=DEFAULT_CONFIG, viewer=None):
    params = BicycleParams()
    model = AgiBotG1MobileBaseBicycleDynamic2Config(
        bicycle_params=params
    ).create_dynamics_model()
    goal = np.asarray(config.goal, dtype=float)
    x0 = np.zeros(6)
    dt = config.bicycle_dt
    steps = config.bicycle_steps
    acceleration_limit = model.robot_cfg.ControlLimit[
        model.robot_cfg.Control.aLinearX
    ]

    def controller(x, _t, _k):
        X, Y, velocity_x, velocity_y, yaw_rate, yaw = x
        desired_yaw = np.arctan2(goal[1] - Y, goal[0] - X)
        yaw_error = (desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
        rotation_world_to_body = np.array(
            [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]], dtype=float
        )
        error_body = rotation_world_to_body @ (goal - np.array([X, Y], dtype=float))
        longitudinal_velocity = max(
            abs(velocity_x), params.minimum_forward_speed
        ) * np.sign(velocity_x + (velocity_x == 0.0))
        steering = yaw_error + np.arctan2(
            config.bicycle_cross_track_gain * error_body[1],
            longitudinal_velocity,
        )
        steering = float(
            np.clip(steering, -params.steering_limit, params.steering_limit)
        )
        distance = float(np.linalg.norm(goal - np.array([X, Y], dtype=float)))
        reference_velocity = min(
            config.bicycle_max_reference_speed,
            config.bicycle_goal_speed_gain * distance,
        )
        acceleration = float(
            np.clip(
                config.bicycle_velocity_gain
                * (reference_velocity - velocity_x)
                - yaw_rate * velocity_y,
                -acceleration_limit,
                acceleration_limit,
            )
        )
        return np.array([acceleration, steering], dtype=float)

    t, states, controls = _rollout(
        model.robot_cfg,
        controller,
        x0,
        steps,
        dt,
        integrator="RK4",
        stop=lambda x, _t: np.linalg.norm(x[:2] - goal) <= 0.1,
    )
    write_csv(
        save_dir / "bicycle_dynamic_states.csv",
        ("t", "X", "Y", "v_x", "v_y", "r", "psi"),
        np.column_stack([t, states]),
    )
    write_csv(
        save_dir / "bicycle_dynamic_controls.csv",
        ("t", "a_x", "delta"),
        np.column_stack([t[:-1], controls]),
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot(states[:, 0], states[:, 1], "k", label="path")
    ax.plot(goal[0], goal[1], "r*", markersize=8, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Dynamic Bicycle Path")
    fig.tight_layout()
    fig.savefig(save_dir / "bicycle_dynamic_path.png", dpi=160)
    plt.close(fig)
    if viewer is not None:
        viewer.replay(
            states[:, [0, 1, 5]],
            sample_dt=dt,
            goal=goal,
            dynamics_model=model,
            integrator="RK4",
            experiment="dynamic bicycle",
        )
    return {"samples": len(t), "final_state": states[-1], "goal_error": np.linalg.norm(states[-1, :2] - goal)}


def _rollout(
    robot_cfg,
    controller,
    initial_state,
    steps,
    dt,
    *,
    integrator,
    stop=None,
    state_dof_names=None,
    control_names=None,
):
    plant = NumericalPlant(
        robot_cfg,
        initial_state,
        dt=dt,
        integrator=integrator,
        max_steps=steps,
        state_dof_names=state_dof_names,
        control_names=control_names,
    )
    trajectory = rollout_model(
        plant,
        controller,
        max_steps=steps,
        stop=(
            None
            if stop is None
            else lambda state, time, _step: stop(state, time)
        ),
    )
    return trajectory.times, trajectory.states, trajectory.controls


def _continuous_reference(model, controller, initial_state, t_final, sample_dt):
    """Fine-step model reference for the lecture comparison."""

    substeps_per_sample = 100
    integration_dt = float(sample_dt) / substeps_per_sample
    integration_steps = round(float(t_final) / integration_dt)
    plant = NumericalPlant(
        model,
        initial_state,
        dt=integration_dt,
        integrator="RK4",
        max_steps=integration_steps,
    )
    trajectory = rollout_model(
        plant,
        controller,
        max_steps=integration_steps,
    )
    sample_indices = np.arange(0, integration_steps + 1, substeps_per_sample)
    control_indices = sample_indices[:-1]
    return (
        trajectory.times[sample_indices],
        trajectory.states[sample_indices],
        trajectory.controls[control_indices],
    )


if __name__ == "__main__":
    run()
