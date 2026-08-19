from pathlib import Path

import numpy as np

from spark_pipeline import AgiBotG1TeleopPipelineConfig as PipelineConfig
from spark_robot import AgiBotG1MobileBaseDynamic2Config

from shared.artifacts import data_dir
from shared.control_pipeline import ControlPipeline, policy_config_components
from .config import DEFAULT_CONFIG, Lecture10MPCConfig


DATA_OUTPUT_DIR = data_dir("lecture10")

X0 = np.array([10.0, 0.0], dtype=float)
GOAL_STATE = np.zeros(2, dtype=float)
DT = 0.5
HORIZON_STEPS = 10
EXECUTION_STEPS = 30

POLICY_CASES = {
    "linear_mpc": "LinearMPCPolicy",
    "input_constrained_mpc": "InputConstrainedMPCPolicy",
    "constrained_mpc": "ConstrainedMPCPolicy",
}


def _reset_dof_pos(x0):
    robot_cfg = AgiBotG1MobileBaseDynamic2Config()
    dof_pos = np.array([robot_cfg.DefaultDoFVal[dof] for dof in robot_cfg.DoFs], dtype=float)
    dof_pos[int(robot_cfg.DoFs.LinearX)] = float(x0[0])
    dof_pos[int(robot_cfg.DoFs.LinearY)] = 0.0
    dof_pos[int(robot_cfg.DoFs.RotYaw)] = 0.0
    return dof_pos


def config_task_module(cfg: PipelineConfig, **kwargs):
    cfg.env.task.enable_ros = False
    cfg.env.task.mode = "Velocity"
    cfg.env.task.num_obstacle_task = 0
    cfg.env.task.arm_goal_enable = False
    cfg.env.task.use_dual_arm = False
    cfg.env.task.base_goal_enable = False
    cfg.env.task.reset_dof_pos = kwargs.get("reset_dof_pos", _reset_dof_pos(kwargs.get("x0", X0)))
    return cfg


def config_agent_module(cfg: PipelineConfig, **kwargs):
    agent_dt = float(kwargs.get("agent_dt", 0.01))
    cfg.env.agent.class_name = "AgiBotG1MobileBaseAgent"
    cfg.env.agent.use_sim_dynamics = False
    cfg.env.agent.enable_viewer = kwargs.get("enable_viewer", False)
    cfg.env.agent.viewer_show_simulation_info = kwargs.get(
        "viewer_show_simulation_info", True
    )
    cfg.env.agent.enable_keyboard_control = False
    cfg.env.agent.enable_camera = False
    cfg.env.agent.real_time = kwargs.get("real_time", cfg.env.agent.enable_viewer)
    cfg.env.agent.dt = agent_dt
    cfg.env.agent.control_decimation = int(kwargs.get("control_decimation", round(DT / agent_dt)))
    cfg.env.agent.model_integrator = kwargs.get("model_integrator", "ZOH")
    cfg.env.agent.obstacle_debug["num_obstacle"] = 0
    return cfg


def config_policy_module(cfg: PipelineConfig, **kwargs):
    x0 = np.asarray(kwargs.get("x0", X0), dtype=float).reshape(2)
    case_name = kwargs.get("case_name", "constrained_mpc")
    policy, nominal_controller, _ = policy_config_components(cfg)
    policy.clip_action_to_control_limits = False
    nominal_controller.class_name = POLICY_CASES[case_name]
    nominal_controller.state_dim = 2
    nominal_controller.control_dim = 1
    nominal_controller.dt = kwargs.get("dt", DT)
    nominal_controller.horizon_steps = kwargs.get("horizon_steps", HORIZON_STEPS)
    nominal_controller.discretization = "ZOH"
    nominal_controller.integrator = "direct"
    nominal_controller.execution_mode = "feedback"
    nominal_controller.state_feedback_mode = "internal_model"
    nominal_controller.command_mode = "direct"
    nominal_controller.clip_robot_action = False
    nominal_controller.initial_state = x0.tolist()
    nominal_controller.goal_state = kwargs.get("goal_state", GOAL_STATE.tolist())
    nominal_controller.Q = kwargs.get("Q", [1.0, 0.0])
    nominal_controller.R = kwargs.get("R", [1.0])
    nominal_controller.S = kwargs.get("S", [10.0, 10.0])
    nominal_controller.state_names = ["x1", "v1"]
    nominal_controller.policy_control_names = ["u1"]
    nominal_controller.position_dof_names = ["LinearX"]
    nominal_controller.velocity_dof_names = ["LinearX"]
    nominal_controller.control_names = ["aLinearX"]
    nominal_controller.zero_control_names = ["aLinearY", "aRotYaw"]

    if case_name in ("input_constrained_mpc", "constrained_mpc"):
        nominal_controller.u_min = kwargs.get("u_min", [-1.0])
        nominal_controller.u_max = kwargs.get("u_max", [1.0])
    if case_name == "constrained_mpc":
        nominal_controller.x_min = kwargs.get("x_min", [-10.0, -2.0])
        nominal_controller.x_max = kwargs.get("x_max", [10.0, 5.0])
        nominal_controller.qp_ftol = kwargs.get("qp_ftol", 1e-10)
        nominal_controller.qp_max_iter = kwargs.get("qp_max_iter", 10000)
    return cfg


def config_safety_module(cfg: PipelineConfig, **kwargs):
    _, _, safe_controller = policy_config_components(cfg)
    safe_controller.safe_algo.class_name = "ByPassSafeControl"
    safe_controller.safety_index.class_name = "SecondOrderCollisionSafetyIndex"
    safe_controller.safety_index.phi_n = 1.0
    safe_controller.safety_index.phi_k = 1.0
    safe_controller.safety_index.enable_self_collision = False
    safe_controller.safety_index.min_distance["environment"] = 0.0
    safe_controller.safety_index.min_distance["self"] = 0.0
    return cfg


def config_pipeline(cfg: PipelineConfig, **kwargs):
    cfg.robot.cfg.class_name = "AgiBotG1MobileBaseDynamic2Config"
    cfg.max_num_steps = kwargs.get("max_num_steps", EXECUTION_STEPS)
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


def make_config(config: Lecture10MPCConfig = DEFAULT_CONFIG, **kwargs):
    defaults = {
        "x0": config.initial_state,
        "goal_state": config.goal_state,
        "dt": config.dt,
        "horizon_steps": config.horizon_steps,
        "model_integrator": config.integrator,
        "max_num_steps": config.execution_steps,
        "case_name": config.default_case,
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


def _reduced_state(agent_feedback, robot_cfg, action_info=None):
    dofs = robot_cfg.DoFs.__members__
    dof_pos = np.asarray(agent_feedback["dof_pos_fbk"], dtype=float).reshape(-1)
    if action_info is not None and "policy_state" in action_info:
        state = np.asarray(action_info["policy_state"], dtype=float).reshape(2).copy()
    else:
        state = X0.copy()
    state[0] = dof_pos[int(dofs["LinearX"])]
    return state


def _policy_control(action, robot_cfg, action_info=None):
    if action_info is not None and "policy_control" in action_info:
        return np.asarray(action_info["policy_control"], dtype=float).reshape(1)
    controls = robot_cfg.Control.__members__
    return np.array([action[int(controls["aLinearX"])]], dtype=float)


def run_case(case_name, **kwargs):
    config = kwargs.pop("config", DEFAULT_CONFIG)
    cfg = make_config(config=config, case_name=case_name, **kwargs)
    _, nominal_controller, _ = policy_config_components(cfg)
    pipeline = ControlPipeline(
        cfg,
        save_dir=kwargs.get("save_dir", DATA_OUTPUT_DIR / case_name),
        state_extractor=_reduced_state,
        action_extractor=_policy_control,
        state_names=("x1", "v1"),
        action_names=("u1",),
        goal_state=nominal_controller.goal_state,
        plot_title=f"Agibot G1 {case_name}",
        plot_filename="state_plot.png",
        experiment_config=config,
        experiment_metadata={"case": case_name},
    )
    return pipeline.run(save_path=kwargs.get("save_path", None))


def run(config: Lecture10MPCConfig = DEFAULT_CONFIG, **kwargs):
    case_name = kwargs.pop("case_name", None)
    if case_name is not None:
        return run_case(case_name, config=config, **kwargs)
    return {name: run_case(name, config=config, **kwargs) for name in POLICY_CASES}


if __name__ == "__main__":
    run()
