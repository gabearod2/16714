from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.utils.control_math import infinite_lqr
from spark_robot import LinearDiscreteDynamicsConfig
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from shared.rl import (
    LinearGaussianActor,
    ScalarQuadraticCritic,
    run_actor_critic,
    run_reinforce,
)
from .config import DEFAULT_CONFIG, Lecture23Config


DATA_OUTPUT_DIR = data_dir("lecture23")


def run(
    save_dir=DATA_OUTPUT_DIR,
    seed: int | None = None,
    config: Lecture23Config = DEFAULT_CONFIG,
):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    A, B, Q, R = config.A, config.B, config.Q, config.R
    x0, tolerance = config.initial_state, config.tolerance
    kmax, dt, n_ep = config.horizon, config.dt, config.episodes
    K, _P = infinite_lqr(np.array([[A]]), np.array([[B]]), np.array([[Q]]), np.array([[R]]))

    def stage_cost(x, u):
        return float((Q * x[0] ** 2 + R * u[0] ** 2) / 2.0)

    plant = NumericalPlant(
        LinearDiscreteDynamicsConfig([[A]], [[B]]),
        [x0],
        dt=dt,
        max_steps=kmax,
    )
    lqr_trajectory = rollout_model(
        plant,
        lambda state, _time, _step: -K @ state,
        max_steps=kmax,
        stop=lambda state, _time, _step: np.linalg.norm(state) < tolerance,
    )
    x_lqr = lqr_trajectory.states

    results = {}
    for offset, alg in enumerate(("REINFORCE", "REINFORCEbl", "ActorCritic")):
        actor = LinearGaussianActor(
            mu=config.initial_mu,
            sigma=config.initial_sigma,
            mu_bounds=config.mu_bounds,
            sigma_bounds=config.sigma_bounds,
        )
        critic = ScalarQuadraticCritic(config.initial_value_weight)
        rng = np.random.default_rng(config.seed + offset)
        rms = np.zeros(n_ep, dtype=float)
        std = np.zeros(n_ep, dtype=float)
        last_states = None
        for episode in range(n_ep):
            learning_plant = NumericalPlant(
                LinearDiscreteDynamicsConfig([[A]], [[B]]),
                [x0],
                dt=dt,
                max_steps=kmax,
            )
            stop = lambda state, step: np.linalg.norm(state) < tolerance
            if alg == "ActorCritic":
                trajectory = run_actor_critic(
                    learning_plant,
                    actor,
                    critic,
                    rng,
                    stage_cost,
                    max_steps=kmax,
                    stop=stop,
                    actor_step=config.actor_step,
                    critic_step=config.critic_step,
                )
            else:
                trajectory = run_reinforce(
                    learning_plant,
                    actor,
                    critic,
                    rng,
                    stage_cost,
                    max_steps=kmax,
                    stop=stop,
                    actor_step=config.actor_step,
                    use_baseline=alg == "REINFORCEbl",
                    critic_step=config.critic_step,
                )
            last_states = trajectory.states
            rms[episode] = abs(actor.mu + K[0, 0])
            std[episode] = actor.sigma
        key = alg.lower()
        results[key] = {"rms": rms, "std": std, "states": last_states}

    write_csv(
        save_dir / "policy_gradient_metrics.csv",
        ("episode", "reinforce", "reinforce_bl", "actor_critic", "reinforce_std", "reinforce_bl_std", "actor_critic_std"),
        np.column_stack(
            [
                np.arange(n_ep),
                results["reinforce"]["rms"],
                results["reinforcebl"]["rms"],
                results["actorcritic"]["rms"],
                results["reinforce"]["std"],
                results["reinforcebl"]["std"],
                results["actorcritic"]["std"],
            ]
        ),
    )
    write_csv(save_dir / "lqr_states.csv", ("k", "x"), np.column_stack([np.arange(x_lqr.shape[0]), x_lqr[:, 0]]))
    for key, value in results.items():
        write_csv(save_dir / f"{key}_final_states.csv", ("k", "x"), np.column_stack([np.arange(value["states"].shape[0]), value["states"][:, 0]]))
    summary = {
        "final_rms_reinforce": results["reinforce"]["rms"][-1],
        "final_rms_reinforce_bl": results["reinforcebl"]["rms"][-1],
        "final_rms_actor_critic": results["actorcritic"]["rms"][-1],
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, results, x_lqr)
    return summary


def _plot(save_dir, results, x_lqr):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=False)
    axes[0].plot(results["reinforce"]["rms"], "b", label="reinforce")
    axes[0].plot(results["reinforcebl"]["rms"], "r", label="reinforce baseline")
    axes[0].plot(results["actorcritic"]["rms"], "k", label="actor critic")
    axes[0].set_title("Policy Parameter Error")
    axes[1].plot(x_lqr[:, 0], "--k", label="lqr")
    for key, value in results.items():
        axes[1].plot(value["states"][:, 0], label=key)
    axes[1].set_title("Final Episode States")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "actor_critic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
