"""Trace-collecting SPARK pipeline used by several robot-control lectures."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import inspect
import time

import numpy as np

from spark_pipeline.base.base_goal_pipeline import BaseGoalPipeline

from shared.artifacts import write_json


def policy_config_components(cfg):
    """Return composition, nominal, and safety configs across SPARK layouts."""
    composition = getattr(cfg, "policy", None)
    if composition is not None and hasattr(composition, "nominal_controller"):
        return composition, composition.nominal_controller, composition.safe_controller

    composition = cfg.algo
    return composition, composition.policy, composition.safe_controller


class ControlPipeline(BaseGoalPipeline):
    """Example-side pipeline for lecture control rollouts.

    The execution loop intentionally follows BaseGoalPipeline.run(): the policy
    returns one executable action per act() call, and this class only adds trace
    collection for planned and executed trajectories.
    """

    def __init__(
        self,
        cfg,
        *,
        save_dir,
        state_extractor,
        action_extractor,
        plan_extractor=None,
        state_names=None,
        action_names=None,
        goal_state=None,
        plot_title="Control Pipeline",
        plot_filename="control_state_plot.png",
        collect_plan_each_step=False,
        experiment_config=None,
        experiment_metadata=None,
    ):
        super().__init__(cfg)
        self.trace_save_dir = Path(save_dir)
        self.state_extractor = state_extractor
        self.action_extractor = action_extractor
        self.plan_extractor = plan_extractor
        self.state_names = tuple(state_names or ())
        self.action_names = tuple(action_names or ())
        self.goal_state = None if goal_state is None else np.asarray(goal_state, dtype=float).reshape(-1)
        self.plot_title = plot_title
        self.plot_filename = plot_filename
        self.collect_plan_each_step = collect_plan_each_step
        self.experiment_config = experiment_config
        self.experiment_metadata = dict(experiment_metadata or {})

        self.planned_traces = []
        self.planned_trace = None
        self.executed_state_t = []
        self.executed_states = []
        self.executed_action_t = []
        self.executed_actions = []
        self._trace_saved = False

    @property
    def control_dt(self):
        return float(self.cfg.env.agent.dt) * int(self.cfg.env.agent.control_decimation)

    def _state_header(self):
        return ",".join(("t",) + self.state_names)

    def _action_header(self):
        return ",".join(("t",) + self.action_names)

    def _extract_state(self, agent_feedback, action_info=None):
        return np.asarray(
            self._call_extractor(self.state_extractor, agent_feedback, self.robot_cfg, action_info),
            dtype=float,
        ).reshape(-1)

    def _extract_action(self, action, action_info=None):
        return np.asarray(
            self._call_extractor(self.action_extractor, action, self.robot_cfg, action_info),
            dtype=float,
        ).reshape(-1)

    @staticmethod
    def _call_extractor(extractor, *args):
        signature = inspect.signature(extractor)
        params = list(signature.parameters.values())
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
        if has_varargs or len(params) >= len(args):
            return extractor(*args)
        return extractor(*args[: len(params)])

    def _collect_planned_trace(self, action_info):
        if self.planned_trace is not None and not self.collect_plan_each_step:
            return

        if self.plan_extractor is None:
            trace = self._extract_policy_plan(action_info)
        else:
            trace = self.plan_extractor(action_info, self)
        if trace is None:
            return

        state_t, states, action_t, actions = trace
        trace_data = {
            "step": self.pipeline_step,
            "state_t": np.asarray(state_t, dtype=float).reshape(-1),
            "states": np.asarray(states, dtype=float),
            "action_t": np.asarray(action_t, dtype=float).reshape(-1),
            "actions": np.asarray(actions, dtype=float),
        }
        self.planned_traces.append(trace_data)
        if self.planned_trace is None:
            self.planned_trace = trace_data

    def _extract_policy_plan(self, action_info):
        plan = action_info.get("policy_plan")
        if plan is None:
            return None

        states = np.asarray(plan["states"], dtype=float)
        actions = np.asarray(plan["controls"], dtype=float)
        state_t = np.asarray(
            plan.get("state_t", np.arange(states.shape[0], dtype=float) * self.control_dt),
            dtype=float,
        ).reshape(-1)
        action_t = np.asarray(
            plan.get("control_t", np.arange(actions.shape[0], dtype=float) * self.control_dt),
            dtype=float,
        ).reshape(-1)

        if not self.state_names and "state_names" in plan:
            self.state_names = tuple(plan["state_names"])
        if not self.action_names and "control_names" in plan:
            self.action_names = tuple(plan["control_names"])

        return state_t, states, action_t, actions

    def _collect_executed_state(self, agent_feedback, action_info=None):
        self.executed_state_t.append(self.pipeline_step * self.control_dt)
        self.executed_states.append(self._extract_state(agent_feedback, action_info))

    def _collect_executed_action(self, action, action_info=None):
        self.executed_action_t.append(self.pipeline_step * self.control_dt)
        self.executed_actions.append(self._extract_action(action, action_info))

    def run(self, save_path=None):
        self.setup_logging(save_path)

        agent_feedback, task_info = self.env.reset()
        self.num_reset = 1
        self._collect_executed_state(agent_feedback)

        action, action_info = self.policy.act(agent_feedback, task_info)
        self._collect_planned_trace(action_info)

        time_start = time.time()
        try:
            while self.pipeline_step < self.max_num_steps or self.max_num_steps < 0:
                start_t = time.time()

                if task_info["done"]:
                    print("Pipeline done")
                    if self.max_num_reset != -1 and self.num_reset >= self.max_num_reset:
                        break

                    agent_feedback, task_info = self.env.reset()
                    self.num_reset += 1
                    self._collect_executed_state(agent_feedback)
                    action, action_info = self.policy.act(agent_feedback, task_info)
                    self._collect_planned_trace(action_info)

                self._collect_executed_action(action, action_info)

                agent_feedback, task_info = self.env.step(action, action_info)
                action, action_info = self.policy.act(agent_feedback, task_info)
                self._collect_planned_trace(action_info)

                step_t = time.time() - start_t
                if getattr(self.cfg.env.agent, "real_time", True) and step_t < self.control_dt:
                    time.sleep(self.control_dt - step_t)
                self.loop_time = time.time() - start_t

                self.pipeline_step += 1
                self.post_physics_step(agent_feedback, task_info, action_info)
                self._collect_executed_state(agent_feedback, action_info)

            print("Pipeline finished in ", time.time() - time_start, " seconds.")
        finally:
            self.save_results()
            self.env.agent.close_viewer()
            if self.cfg.enable_plotter:
                print("Terminating plotter process")
                self.plotter_process.terminate()
                self.plotter_process.wait()

        return self.trace_data()

    def trace_data(self):
        planned = self.planned_trace
        return {
            "pipeline": self,
            "planned": planned,
            "state_t": None if planned is None else planned["state_t"],
            "states": None if planned is None else planned["states"],
            "control_t": None if planned is None else planned["action_t"],
            "controls": None if planned is None else planned["actions"],
            "executed_state_t": np.asarray(self.executed_state_t, dtype=float),
            "executed_states": np.asarray(self.executed_states, dtype=float),
            "executed_control_t": np.asarray(self.executed_action_t, dtype=float),
            "executed_controls": np.asarray(self.executed_actions, dtype=float),
        }

    def save_results(self):
        super().save_results()
        if self._trace_saved:
            return
        self._save_trace_outputs()
        self._trace_saved = True

    def _save_trace_outputs(self):
        self.trace_save_dir.mkdir(parents=True, exist_ok=True)

        if self.planned_trace is not None:
            self._save_trace(
                "",
                self.planned_trace["state_t"],
                self.planned_trace["states"],
                self.planned_trace["action_t"],
                self.planned_trace["actions"],
            )

        self._save_trace(
            "executed",
            np.asarray(self.executed_state_t, dtype=float),
            np.asarray(self.executed_states, dtype=float),
            np.asarray(self.executed_action_t, dtype=float),
            np.asarray(self.executed_actions, dtype=float),
        )
        self._save_trace_plot(self.trace_save_dir / self.plot_filename)
        self._save_validation_metadata()

        print(f"Saved control artifacts to: {self.trace_save_dir.resolve()}")
        if self.planned_trace is not None:
            state_t = self.planned_trace["state_t"]
            action_t = self.planned_trace["action_t"]
            print(
                "Saved planned trajectory samples: "
                f"{len(state_t)} states, {len(action_t)} controls, "
                f"t=[{state_t[0]:.3g}, {state_t[-1]:.3g}]"
            )

    def _save_validation_metadata(self):
        if self.experiment_config is not None:
            config = (
                asdict(self.experiment_config)
                if is_dataclass(self.experiment_config)
                else self.experiment_config
            )
            write_json(
                self.trace_save_dir / "config.json",
                {"config": config, **self.experiment_metadata},
            )

        executed_states = np.asarray(self.executed_states, dtype=float)
        executed_actions = np.asarray(self.executed_actions, dtype=float)
        summary = {
            "num_executed_states": int(len(executed_states)),
            "num_executed_controls": int(len(executed_actions)),
        }
        if len(executed_states):
            summary["final_executed_state"] = executed_states[-1]
            if self.goal_state is not None and executed_states.shape[1] == self.goal_state.size:
                summary["final_goal_error_norm"] = float(
                    np.linalg.norm(executed_states[-1] - self.goal_state)
                )
        if len(executed_actions):
            summary["final_executed_control"] = executed_actions[-1]
        if self.planned_trace is not None:
            summary["num_planned_states"] = int(len(self.planned_trace["states"]))
            summary["num_planned_controls"] = int(len(self.planned_trace["actions"]))
        write_json(self.trace_save_dir / "summary.json", summary)

    def _save_trace(self, prefix, state_t, states, action_t, actions):
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        stem = f"{prefix}_" if prefix else ""

        np.savetxt(
            self.trace_save_dir / f"{stem}states.csv",
            np.column_stack([state_t, states]),
            delimiter=",",
            header=self._state_header(),
            comments="",
        )
        np.savetxt(
            self.trace_save_dir / f"{stem}controls.csv",
            np.column_stack([action_t, actions]),
            delimiter=",",
            header=self._action_header(),
            comments="",
        )

    def _save_trace_plot(self, path):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        planned = self.planned_trace
        executed_state_t = np.asarray(self.executed_state_t, dtype=float)
        executed_states = np.asarray(self.executed_states, dtype=float)
        executed_action_t = np.asarray(self.executed_action_t, dtype=float)
        executed_actions = np.asarray(self.executed_actions, dtype=float)

        fig = plt.figure(figsize=(7.6, 4.6))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 1.28),
            height_ratios=(1.0, 1.0),
            hspace=0.42,
            wspace=0.34,
        )
        ax_path = fig.add_subplot(gs[:, 0])
        ax_state = fig.add_subplot(gs[0, 1])
        ax_action = fig.add_subplot(gs[1, 1])
        fig.suptitle(self.plot_title)

        if planned is not None:
            states = planned["states"]
            state_t = planned["state_t"]
            actions = planned["actions"]
            action_t = planned["action_t"]

            ax_path.plot(states[:, 0], states[:, 1], color="tab:blue", linewidth=2, label="planned")
            ax_path.plot(states[0, 0], states[0, 1], "ko", markersize=4, label="start")
            ax_path.plot(states[-1, 0], states[-1, 1], "o", color="tab:red", markersize=4, label="planned final")
            self._plot_signal(ax_state, state_t, states, self.state_names, solid=True)
            self._plot_signal(ax_action, action_t, actions, self.action_names, solid=True)

        if executed_states.size > 0:
            ax_path.plot(
                executed_states[:, 0],
                executed_states[:, 1],
                color="tab:orange",
                linestyle="--",
                linewidth=1.5,
                label="executed",
            )
            ax_path.plot(
                executed_states[-1, 0],
                executed_states[-1, 1],
                "s",
                color="tab:orange",
                markersize=4,
                label="executed final",
            )
            self._plot_signal(ax_state, executed_state_t, executed_states, self.state_names, solid=False)

        if executed_actions.size > 0:
            self._plot_signal(ax_action, executed_action_t, executed_actions, self.action_names, solid=False)

        if self.goal_state is not None and self.goal_state.size >= 2:
            ax_path.plot(self.goal_state[0], self.goal_state[1], "r*", markersize=8, label="goal")

        ax_path.set_title("Base Path")
        ax_path.set_xlabel(self.state_names[0] if self.state_names else "state 0")
        ax_path.set_ylabel(self.state_names[1] if len(self.state_names) > 1 else "state 1")
        ax_path.set_aspect("equal", adjustable="datalim")
        ax_state.set_title("State")
        ax_state.set_xlabel("t [s]")
        ax_state.set_ylabel("state")
        ax_action.set_title("Control")
        ax_action.set_xlabel("t [s]")
        ax_action.set_ylabel("control")

        for ax in (ax_path, ax_state, ax_action):
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.88)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _plot_signal(ax, t, values, names, solid):
        colors = (
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        )
        linestyle = "-" if solid else "--"
        suffix = "planned" if solid else "executed"
        for idx in range(values.shape[1]):
            name = names[idx] if idx < len(names) else f"u{idx}"
            ax.plot(
                t,
                values[:, idx],
                color=colors[idx % len(colors)],
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{name} {suffix}",
            )
