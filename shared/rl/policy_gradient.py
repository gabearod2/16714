"""Course policy-gradient update rules, independent of plant dynamics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .actors import LinearGaussianActor, bounded_step
from .critics import ScalarQuadraticCritic
from .episode import collect_episode
from .types import Episode


def run_reinforce(
    plant,
    actor: LinearGaussianActor,
    critic: ScalarQuadraticCritic,
    rng: np.random.Generator,
    stage_cost: Callable[[np.ndarray, np.ndarray], float],
    *,
    max_steps: int,
    stop,
    actor_step: float,
    use_baseline: bool,
    critic_step: float = 0.0,
) -> Episode:
    episode = collect_episode(
        plant,
        lambda state: actor.sample(state, rng),
        stage_cost,
        max_steps=max_steps,
        stop=stop,
    )
    actor_gradient = np.zeros(2, dtype=float)
    critic_gradient = 0.0
    for step in range(episode.length):
        cost_to_go = float(np.sum(episode.costs[step:]))
        baseline = critic.value(episode.states[step]) if use_baseline else 0.0
        advantage = cost_to_go - baseline
        actor_gradient += bounded_step(
            actor_step * (advantage if use_baseline else cost_to_go),
            actor.grad_log_prob(episode.states[step], episode.actions[step]),
        )
        if use_baseline:
            critic_gradient += float(
                bounded_step(
                    critic_step * advantage,
                    np.array([critic.gradient(episode.states[step])]),
                )[0]
            )
    actor.apply_cost_gradient(actor_gradient)
    if use_baseline:
        critic.weight += critic_gradient
    return episode


def run_actor_critic(
    plant,
    actor: LinearGaussianActor,
    critic: ScalarQuadraticCritic,
    rng: np.random.Generator,
    stage_cost: Callable[[np.ndarray, np.ndarray], float],
    *,
    max_steps: int,
    stop,
    actor_step: float,
    critic_step: float,
) -> Episode:
    state = np.asarray(plant.reset(), dtype=float).reshape(-1)
    states = [state.copy()]
    times = [plant.time]
    actions = []
    costs = []
    # Preserve the lecture's original actor-critic stopping convention, which
    # terminated when ``step > max_steps`` and therefore performs the endpoint
    # update as well.  Episodic REINFORCE uses the ordinary max-step collector.
    for step in range(int(max_steps) + 1):
        if stop is not None and stop(state, step):
            break
        action = actor.sample(state, rng)
        new_state = np.asarray(plant.step(action), dtype=float).reshape(-1)
        cost = float(stage_cost(state, action))
        td_error = cost + critic.value(new_state) - critic.value(state)
        actor.apply_cost_gradient(
            bounded_step(
                actor_step * td_error,
                actor.grad_log_prob(state, action),
            )
        )
        critic.weight += float(
            bounded_step(
                critic_step * td_error,
                np.array([critic.gradient(state)]),
            )[0]
        )
        actions.append(action.copy())
        costs.append(cost)
        state = new_state
        states.append(state.copy())
        times.append(plant.time)
    return Episode(
        times=np.asarray(times),
        states=np.vstack(states),
        actions=np.vstack(actions) if actions else np.empty((0, 0)),
        costs=np.asarray(costs),
    )
