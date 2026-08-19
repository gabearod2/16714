"""Course RL episode collection over a course numerical plant."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .types import Episode


def collect_episode(
    plant,
    actor: Callable[[np.ndarray], np.ndarray],
    stage_cost: Callable[[np.ndarray, np.ndarray], float],
    *,
    max_steps: int,
    stop: Callable[[np.ndarray, int], bool] | None = None,
) -> Episode:
    """Collect one episode while the supplied plant owns its dynamics model."""

    state = np.asarray(plant.reset(), dtype=float).reshape(-1)
    states = [state.copy()]
    times = [plant.time]
    actions: list[np.ndarray] = []
    costs: list[float] = []

    for step in range(int(max_steps)):
        if stop is not None and stop(state, step):
            break
        action = np.asarray(actor(state), dtype=float).reshape(-1)
        costs.append(float(stage_cost(state, action)))
        actions.append(action.copy())
        state = np.asarray(plant.step(action), dtype=float).reshape(-1)
        states.append(state.copy())
        times.append(plant.time)
        if plant.done:
            break

    action_dim = actions[0].size if actions else 0
    return Episode(
        times=np.asarray(times, dtype=float),
        states=np.vstack(states),
        actions=(
            np.vstack(actions)
            if actions
            else np.empty((0, action_dim), dtype=float)
        ),
        costs=np.asarray(costs, dtype=float),
    )
