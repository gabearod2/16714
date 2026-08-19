"""Lecture 21 quadratic value representation and update methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.rl import Episode, collect_episode


@dataclass
class QuadraticQFunction:
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).copy()

    @staticmethod
    def features(state, action) -> np.ndarray:
        return np.concatenate(
            [np.asarray(state, dtype=float).reshape(-1), np.asarray(action, dtype=float).reshape(-1)]
        )

    def q_value(self, state, action) -> float:
        features = self.features(state, action)
        return float(features.T @ self.weights @ features / 2.0)

    def gradient(self, state, action) -> np.ndarray:
        features = self.features(state, action)
        return np.outer(features, features) / 2.0

    def value(self, state) -> float:
        state = np.asarray(state, dtype=float).reshape(-1)
        state_dim = state.size
        Wxx = self.weights[:state_dim, :state_dim]
        Wxu = self.weights[:state_dim, state_dim:]
        Wux = self.weights[state_dim:, :state_dim]
        Wuu = self.weights[state_dim:, state_dim:]
        return float(state.T @ (Wxx - Wxu @ np.linalg.inv(Wuu) @ Wux) @ state / 2.0)

    def greedy_action(
        self,
        state,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        state_dim = state.size
        Wuu = self.weights[state_dim:, state_dim:]
        Wux = self.weights[state_dim:, :state_dim]
        Wxu = self.weights[:state_dim, state_dim:]
        action = -np.linalg.solve(Wuu, (Wux + Wxu.T) / 2.0) @ state
        if rng.random() < float(epsilon):
            action = action + (-rng.random((Wuu.shape[0], state_dim))) @ state
        return np.asarray(action, dtype=float).reshape(-1)


def run_value_episode(
    algorithm: str,
    plant,
    q_function: QuadraticQFunction,
    rng: np.random.Generator,
    stage_cost,
    *,
    learning_rate: float,
    epsilon: float,
    max_steps: int,
    stop,
) -> Episode:
    key = algorithm.lower()
    actor = lambda state: q_function.greedy_action(state, epsilon=epsilon, rng=rng)
    if key == "mc":
        episode = collect_episode(
            plant,
            actor,
            stage_cost,
            max_steps=max_steps,
            stop=stop,
        )
        update = np.zeros_like(q_function.weights)
        for step in range(episode.length):
            return_value = float(np.sum(episode.costs[step:]))
            update += float(learning_rate) * (
                return_value - q_function.q_value(episode.states[step], episode.actions[step])
            ) * q_function.gradient(episode.states[step], episode.actions[step])
        q_function.weights += update
        return episode

    state = np.asarray(plant.reset(), dtype=float).reshape(-1)
    states = [state.copy()]
    times = [plant.time]
    actions = []
    costs = []
    action = actor(state) if key == "sarsa" else None
    for step in range(int(max_steps)):
        if stop(state, step):
            break
        if key == "qlearning":
            action = actor(state)
        if action is None:
            raise ValueError(f'Unknown value-learning algorithm "{algorithm}".')
        new_state = np.asarray(plant.step(action), dtype=float).reshape(-1)
        cost = float(stage_cost(state, action))
        if key == "sarsa":
            new_action = actor(new_state)
            target = cost + q_function.q_value(new_state, new_action)
        elif key == "qlearning":
            new_action = None
            target = cost + q_function.value(new_state)
        else:
            raise ValueError(f'Unknown value-learning algorithm "{algorithm}".')
        q_function.weights += float(learning_rate) * (
            target - q_function.q_value(state, action)
        ) * q_function.gradient(state, action)
        actions.append(action.copy())
        costs.append(cost)
        state = new_state
        states.append(state.copy())
        times.append(plant.time)
        action = new_action
        if plant.done:
            break
    return Episode(
        times=np.asarray(times),
        states=np.vstack(states),
        actions=np.vstack(actions) if actions else np.empty((0, 0)),
        costs=np.asarray(costs),
    )
