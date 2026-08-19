"""Course-owned state and rollout helpers for numerical dynamics models."""

from dataclasses import dataclass

import numpy as np

from spark_robot import DynamicsStepContext


@dataclass(frozen=True)
class NumericalTrajectory:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray


class NumericalPlant:
    """Minimal stateful wrapper that advances a SPARK model without a backend."""

    def __init__(
        self,
        robot_config_or_model,
        initial_state,
        *,
        dt,
        integrator=None,
        max_steps=None,
        state_dof_names=None,
        control_names=None,
        seed=None,
    ):
        create_model = getattr(robot_config_or_model, "create_dynamics_model", None)
        self.model = (
            create_model(
                state_dof_names=state_dof_names,
                control_names=control_names,
            )
            if create_model is not None
            else robot_config_or_model
        )
        self.initial_state = np.asarray(initial_state, dtype=float).reshape(
            self.model.state_dim
        )
        self.dt = float(dt)
        self.integrator = integrator or getattr(
            self.model, "default_integrator", "Euler"
        )
        self.max_steps = None if max_steps is None else int(max_steps)
        self.seed = seed
        self.episode_index = -1
        self.state = None
        self.time = 0.0
        self.step_index = 0

    @property
    def done(self):
        return self.max_steps is not None and self.step_index >= self.max_steps

    def reset(self):
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.step_index = 0
        self.episode_index += 1
        self.rng = np.random.default_rng(self.seed)
        return self.state.copy()

    def step(self, control):
        if self.state is None:
            raise RuntimeError("NumericalPlant.reset() must be called before step().")
        context = DynamicsStepContext(
            time=self.time,
            step_index=self.step_index,
            episode_index=self.episode_index,
            rng=self.rng,
        )
        self.state = self.model.step(
            self.state,
            np.asarray(control, dtype=float),
            self.dt,
            self.integrator,
            context=context,
        )
        self.step_index += 1
        self.time += self.dt
        return self.state.copy()


def rollout_model(plant, controller, *, max_steps=None, stop=None):
    """Roll out one course numerical plant with state/time/step callbacks."""

    steps = plant.max_steps if max_steps is None else int(max_steps)
    if steps is None:
        raise ValueError("max_steps is required when the plant has no limit.")
    state = plant.reset()
    times = [plant.time]
    states = [state]
    controls = []
    for step in range(steps):
        if plant.done or (
            stop is not None and stop(state, plant.time, step)
        ):
            break
        control = np.asarray(
            controller(state, plant.time, step), dtype=float
        ).reshape(plant.model.control_dim)
        controls.append(control.copy())
        state = plant.step(control)
        states.append(state)
        times.append(plant.time)
    return NumericalTrajectory(
        times=np.asarray(times, dtype=float),
        states=np.vstack(states),
        controls=(
            np.vstack(controls)
            if controls
            else np.empty((0, plant.model.control_dim), dtype=float)
        ),
    )
