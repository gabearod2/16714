"""Joint- and Cartesian-space control of the SPARK AgiBot G1 right arm."""

from __future__ import annotations

from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_agent import AgiBotG1RightArmAgent
from spark_robot import AgiBotG1RightArmDynamic1Config, AgiBotG1RightArmKinematics
from spark_utils import VizColor

from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture3Config


DATA_OUTPUT_DIR = data_dir("lecture3")
ROBOT_CONFIG_NAME = "AgiBotG1RightArmDynamic1Config"
END_EFFECTOR_FRAME = "R_ee"


class AgiBotRightArmExperiment:
    """Own the SPARK robot model, kinematics, and optional MuJoCo viewer."""

    def __init__(self, config: Lecture3Config):
        self.config = config
        self.robot_cfg = AgiBotG1RightArmDynamic1Config()
        self.kinematics = AgiBotG1RightArmKinematics(self.robot_cfg)
        self.dynamics = self.robot_cfg.create_dynamics_model()
        self.default_q = np.array(
            [self.robot_cfg.DefaultDoFVal[dof] for dof in self.robot_cfg.DoFs],
            dtype=float,
        )
        self.joint_names = tuple(dof.name for dof in self.robot_cfg.DoFs)
        self.control_names = tuple(control.name for control in self.robot_cfg.Control)
        self.control_limits = np.array(
            [self.robot_cfg.ControlLimit[control] for control in self.robot_cfg.Control],
            dtype=float,
        )
        self.joint_lower, self.joint_upper = self._joint_limits()
        self.agent = self._make_agent() if config.enable_viewer else None

    def _joint_limits(self):
        limits = [
            self.robot_cfg.RealMotorPosLimit[
                self.robot_cfg.RealMotors.__members__[dof.name]
            ]
            for dof in self.robot_cfg.DoFs
        ]
        limits = np.asarray(limits, dtype=float)
        return limits[:, 0], limits[:, 1]

    def _make_agent(self):
        return AgiBotG1RightArmAgent(
            self.robot_cfg,
            enable_viewer=True,
            viewer_show_simulation_info=self.config.viewer_show_simulation_info,
            enable_keyboard_control=False,
            enable_camera=False,
            use_sim_dynamics=False,
            dt=self.config.dt,
            control_decimation=1,
            real_time=self.config.real_time,
            obstacle_debug={"num_obstacle": 0},
            viewer_config={
                "camera_lookat": (0.15, -0.10, 0.75),
                "camera_distance": 2.1,
                "camera_azimuth": 145.0,
                "camera_elevation": -20.0,
            },
        )

    def reset_viewer(self, q):
        if self.agent is None:
            return
        self.agent.reset({"reset_dof_pos": np.asarray(q, dtype=float)})
        self.render_viewer()

    def step(self, q, velocity, *, goal_position=None):
        velocity = self.clip_velocity(velocity)
        q_next = self.dynamics.step(q, velocity, self.config.dt, "Euler")
        q_next = np.clip(q_next, self.joint_lower, self.joint_upper)

        if self.agent is not None and self.agent.is_running():
            self.agent.step(velocity, action_info={})
            self.render_viewer(goal_position=goal_position)
        return q_next, velocity

    def render_viewer(self, *, goal_position=None):
        if self.agent is None or not self.agent.is_running():
            return
        self.agent.begin_render_frame()
        feedback = self.agent.get_feedback()
        ee_frame = self.ee_frame(feedback["dof_pos_fbk"])
        base_frame = feedback["robot_base_frame"]
        ee_frame_world = base_frame @ ee_frame
        self.agent.render_coordinate_frame(ee_frame_world, size=0.10)
        if goal_position is not None:
            goal_world = base_frame[:3, :3] @ np.asarray(goal_position) + base_frame[:3, 3]
            self.agent.render_sphere(
                goal_world,
                np.eye(3),
                0.04 * np.ones(3),
                VizColor.goal,
            )
        self.agent.render()

    def clip_velocity(self, velocity):
        return np.clip(
            np.asarray(velocity, dtype=float).reshape(len(self.control_names)),
            -self.control_limits,
            self.control_limits,
        )

    def ee_frame(self, q):
        frames = self.kinematics.forward_kinematics(np.asarray(q, dtype=float))
        return frames[self.robot_cfg.Frames.R_ee].copy()

    def ee_position(self, q):
        return self.ee_frame(q)[:3, 3]

    def position_jacobian(self, q):
        self.kinematics.pre_computation(np.asarray(q, dtype=float))
        return self.kinematics.get_jacobian(self.robot_cfg.Frames.R_ee)[:3, :]

    def close(self):
        if self.agent is not None:
            self.agent.close_viewer()


def run(
    save_dir=DATA_OUTPUT_DIR,
    seed=None,
    config: Lecture3Config = DEFAULT_CONFIG,
    *,
    enable_viewer=None,
    viewer_show_simulation_info=None,
    real_time=None,
):
    if seed is not None:
        config = replace(config, seed=int(seed))
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
    rng = np.random.default_rng(config.seed)
    experiment = AgiBotRightArmExperiment(config)
    try:
        random_summary = _random_joint_motion(save_dir, rng, experiment)
        servo_summary = _joint_vs_cartesian_servo(save_dir, experiment)
    finally:
        experiment.close()

    write_json(save_dir / "config.json", asdict(config))
    summary = {
        "robot_config": ROBOT_CONFIG_NAME,
        "end_effector_frame": END_EFFECTOR_FRAME,
        "num_joints": len(experiment.joint_names),
        "random_joint_motion": random_summary,
        "joint_vs_cartesian_servo": servo_summary,
    }
    write_json(save_dir / "summary.json", summary)
    return summary


def _random_joint_motion(save_dir, rng, experiment):
    config = experiment.config
    q = experiment.default_q.copy()
    experiment.reset_viewer(q)
    states = [q.copy()]
    ee_positions = [experiment.ee_position(q)]
    controls = []

    for _ in range(config.steps):
        velocity = rng.normal(0.0, config.random_velocity_std, len(q))
        q, velocity = experiment.step(q, velocity)
        controls.append(velocity)
        states.append(q.copy())
        ee_positions.append(experiment.ee_position(q))

    states = np.asarray(states)
    controls = np.asarray(controls)
    ee_positions = np.asarray(ee_positions)
    t = np.arange(config.steps + 1) * config.dt
    write_csv(
        save_dir / "random_joint_states.csv",
        ("t",) + experiment.joint_names + ("ee_x", "ee_y", "ee_z"),
        np.column_stack([t, states, ee_positions]),
    )
    write_csv(
        save_dir / "random_joint_controls.csv",
        ("t",) + experiment.control_names,
        np.column_stack([t[:-1], controls]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(t, states)
    axes[0].set_title("AgiBot Right-Arm Joint Motion")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("joint position [rad]")
    axes[0].grid(True)
    axes[1].plot(t, ee_positions)
    axes[1].set_title("AgiBot R_ee Position")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("position [m]")
    axes[1].legend(("x", "y", "z"), loc="best")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "random_joint_motion.png", dpi=160)
    plt.close(fig)
    return {
        "final_joint_displacement_norm": np.linalg.norm(states[-1] - states[0]),
        "end_effector_path_length": _path_length(ee_positions),
    }


def _joint_vs_cartesian_servo(save_dir, experiment):
    config = experiment.config
    q0 = experiment.default_q.copy()
    q_goal = np.clip(
        q0 + np.asarray(config.goal_joint_offset, dtype=float),
        experiment.joint_lower,
        experiment.joint_upper,
    )
    position_goal = experiment.ee_position(q_goal)

    joint = _run_servo(
        experiment,
        q0,
        position_goal,
        lambda q: config.joint_gain * (q_goal - q),
    )
    cartesian = _run_servo(
        experiment,
        q0,
        position_goal,
        lambda q: _cartesian_velocity(experiment, q, position_goal),
    )

    t = np.arange(config.steps + 1) * config.dt
    _write_servo_trace(save_dir, "joint", t, joint, experiment)
    _write_servo_trace(save_dir, "cartesian", t, cartesian, experiment)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(t, np.linalg.norm(joint["states"] - q_goal, axis=1), label="joint servo")
    axes[0].plot(t, np.linalg.norm(cartesian["states"] - q_goal, axis=1), label="Cartesian servo")
    axes[0].set_title("Distance to Selected Joint Goal")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$\|q-q_{goal}\|_2$ [rad]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, np.linalg.norm(joint["positions"] - position_goal, axis=1), label="joint servo")
    axes[1].plot(t, np.linalg.norm(cartesian["positions"] - position_goal, axis=1), label="Cartesian servo")
    axes[1].set_title("AgiBot R_ee Position Error")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$\|p-p_{goal}\|_2$ [m]")
    axes[1].legend(loc="best")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "joint_vs_cartesian.png", dpi=160)
    plt.close(fig)

    return {
        "joint_goal": q_goal,
        "joint_final_joint_error": np.linalg.norm(joint["states"][-1] - q_goal),
        "joint_final_cartesian_error": np.linalg.norm(joint["positions"][-1] - position_goal),
        "cartesian_final_joint_error": np.linalg.norm(cartesian["states"][-1] - q_goal),
        "cartesian_final_cartesian_error": np.linalg.norm(
            cartesian["positions"][-1] - position_goal
        ),
        "cartesian_goal": position_goal,
    }


def _run_servo(experiment, q0, position_goal, controller):
    q = q0.copy()
    experiment.reset_viewer(q)
    states = [q.copy()]
    positions = [experiment.ee_position(q)]
    controls = []
    for _ in range(experiment.config.steps):
        q, velocity = experiment.step(q, controller(q), goal_position=position_goal)
        controls.append(velocity)
        states.append(q.copy())
        positions.append(experiment.ee_position(q))
    return {
        "states": np.asarray(states),
        "positions": np.asarray(positions),
        "controls": np.asarray(controls),
    }


def _cartesian_velocity(experiment, q, position_goal):
    error = position_goal - experiment.ee_position(q)
    jacobian = experiment.position_jacobian(q)
    damping = experiment.config.jacobian_damping
    damped_inverse = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2 * np.eye(3),
        np.eye(3),
    )
    return experiment.config.cartesian_gain * damped_inverse @ error


def _write_servo_trace(save_dir, name, t, trace, experiment):
    write_csv(
        save_dir / f"{name}_servo_states.csv",
        ("t",) + experiment.joint_names + ("ee_x", "ee_y", "ee_z"),
        np.column_stack([t, trace["states"], trace["positions"]]),
    )
    write_csv(
        save_dir / f"{name}_servo_controls.csv",
        ("t",) + experiment.control_names,
        np.column_stack([t[:-1], trace["controls"]]),
    )


def _path_length(points):
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


if __name__ == "__main__":
    run()
