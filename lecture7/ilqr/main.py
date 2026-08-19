from pathlib import Path

import numpy as np

from spark_pipeline import AgiBotG1TeleopPipelineConfig as PipelineConfig
from shared.artifacts import data_dir
from shared.control_pipeline import ControlPipeline, policy_config_components
from .config import DEFAULT_CONFIG, Lecture7ILQRConfig


DATA_OUTPUT_DIR = data_dir("lecture7/ilqr")


def config_task_module(cfg: PipelineConfig, **kwargs):
    cfg.env.task.enable_ros = False
    cfg.env.task.mode = "Velocity"
    cfg.env.task.num_obstacle_task = 0
    cfg.env.task.arm_goal_enable = False
    cfg.env.task.use_dual_arm = False
    cfg.env.task.base_goal_enable = True
    cfg.env.task.base_goal_init = kwargs.get("base_goal_init", [0.5, 0.25, 0.793])
    return cfg


def config_agent_module(cfg: PipelineConfig, **kwargs):
    cfg.env.agent.class_name = "AgiBotG1MobileBaseAgent"
    cfg.env.agent.use_sim_dynamics = False
    cfg.env.agent.enable_viewer = kwargs.get("enable_viewer", True)
    cfg.env.agent.viewer_show_simulation_info = kwargs.get(
        "viewer_show_simulation_info", True
    )
    cfg.env.agent.enable_keyboard_control = False
    cfg.env.agent.enable_camera = False
    cfg.env.agent.real_time = kwargs.get("real_time", cfg.env.agent.enable_viewer)
    cfg.env.agent.dt = kwargs.get("agent_dt", 0.002)
    cfg.env.agent.control_decimation = kwargs.get("control_decimation", 50)
    cfg.env.agent.model_integrator = kwargs.get("model_integrator", "Euler")
    cfg.env.agent.obstacle_debug["num_obstacle"] = 0
    return cfg


def config_policy_module(cfg: PipelineConfig, **kwargs):
    _, nominal_controller, _ = policy_config_components(cfg)
    nominal_controller.class_name = "ILQRPolicy"
    nominal_controller.solve_mode = "plan_once"
    nominal_controller.N = kwargs.get("N", 150)
    nominal_controller.dt = kwargs.get("ilqr_dt", 0.1)
    nominal_controller.integrator = "Euler"
    nominal_controller.initial_nominal = "rollout"
    nominal_controller.Q = kwargs.get("Q", [1.0, 1.0, 0.1])
    nominal_controller.R = kwargs.get("R", [2.0, 2.5])
    nominal_controller.S = kwargs.get("S", [100.0, 100.0, 50.0])
    nominal_controller.u_ref = [0.0, 0.0]
    nominal_controller.alphas = [1.0, 0.5, 0.25, 0.1, 0.05]
    nominal_controller.max_iter = kwargs.get("max_iter", 120)
    nominal_controller.tol_j = kwargs.get("tol_j", 1e-8)
    nominal_controller.lambda0 = 1e-8
    nominal_controller.lambda_max = 1e8
    nominal_controller.goal_state = kwargs.get("goal_state", [0.5, 0.25, 0.3])
    nominal_controller.verbose = kwargs.get("verbose", False)
    return cfg


def config_safety_module(cfg: PipelineConfig, **kwargs):
    _, _, safe_controller = policy_config_components(cfg)
    safe_controller.safe_algo.class_name = "ByPassSafeControl"
    safe_controller.safety_index.class_name = "FirstOrderCollisionSafetyIndex"
    safe_controller.safety_index.enable_self_collision = False
    safe_controller.safety_index.min_distance["environment"] = 0.0
    safe_controller.safety_index.min_distance["self"] = 0.0
    return cfg


def config_pipeline(cfg: PipelineConfig, **kwargs):
    cfg.robot.cfg.class_name = "AgiBotG1MobileBaseUnicycleDynamic1Config"
    cfg.max_num_steps = kwargs.get("max_num_steps", 150)
    cfg.max_num_reset = 1
    cfg.enable_logger = False
    cfg.enable_plotter = False
    cfg.enable_safe_zone_render = False
    cfg.metric_selection.dof_pos = True
    cfg.metric_selection.dof_vel = True
    cfg.metric_selection.dist_goal_base = False
    cfg.metric_selection.dist_goal_arm = False
    cfg.metric_selection.dist_robot_to_env = False
    cfg.metric_selection.dist_self = False
    cfg.metric_selection.trigger_safe_controller = False
    return cfg


def make_config(config: Lecture7ILQRConfig = DEFAULT_CONFIG, **kwargs):
    defaults = {
        "base_goal_init": config.base_goal,
        "goal_state": config.goal_state,
        "N": config.horizon_steps,
        "ilqr_dt": config.dt,
        "model_integrator": config.integrator,
        "max_iter": config.max_iterations,
        "max_num_steps": config.execution_steps,
        "enable_viewer": config.enable_viewer,
        "viewer_show_simulation_info": config.viewer_show_simulation_info,
    }
    defaults.update(kwargs)
    kwargs = defaults
    cfg = PipelineConfig()
    cfg = config_pipeline(cfg, **kwargs)
    cfg = config_task_module(cfg, **kwargs)
    cfg = config_agent_module(cfg, **kwargs)
    cfg = config_policy_module(cfg, **kwargs)
    cfg = config_safety_module(cfg, **kwargs)
    return cfg


def _reduced_state(agent_feedback, robot_cfg):
    dofs = robot_cfg.DoFs.__members__
    dof_pos = np.asarray(agent_feedback["dof_pos_fbk"], dtype=float).reshape(-1)
    return np.array(
        [
            dof_pos[int(dofs["LinearX"])],
            dof_pos[int(dofs["LinearY"])],
            dof_pos[int(dofs["RotYaw"])],
        ],
        dtype=float,
    )


def _reduced_control(action, robot_cfg):
    controls = robot_cfg.Control.__members__
    return np.array(
        [
            action[int(controls["vLinearX"])],
            action[int(controls["vRotYaw"])],
        ],
        dtype=float,
    )


def run(config: Lecture7ILQRConfig = DEFAULT_CONFIG, **kwargs):
    cfg = make_config(config=config, **kwargs)
    _, nominal_controller, _ = policy_config_components(cfg)
    pipeline = ControlPipeline(
        cfg,
        save_dir=DATA_OUTPUT_DIR,
        state_extractor=_reduced_state,
        action_extractor=_reduced_control,
        state_names=("x", "y", "theta"),
        action_names=("v", "omega"),
        goal_state=nominal_controller.goal_state,
        plot_title="Agibot G1 iLQR",
        plot_filename="ilqr_state_plot.png",
        experiment_config=config,
    )
    return pipeline.run(save_path=kwargs.get("save_path", None))

if __name__ == "__main__":
    run()
