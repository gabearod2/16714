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
from .config import DEFAULT_CONFIG, Lecture21Config
from .value_learning import QuadraticQFunction, run_value_episode


DATA_OUTPUT_DIR = data_dir("lecture21")


def run(
    save_dir=DATA_OUTPUT_DIR,
    seed: int | None = None,
    config: Lecture21Config = DEFAULT_CONFIG,
):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    A, B, Q, R = config.A, config.B, config.Q, config.R
    tolerance = config.tolerance
    kmax, dt, x0 = config.horizon, config.dt, config.initial_state
    n_ep, alpha = config.episodes, config.learning_rate
    W0 = np.asarray(config.initial_q_weights, dtype=float)
    K, P = infinite_lqr(np.array([[A]]), np.array([[B]]), np.array([[Q]]), np.array([[R]]))
    W_gt = np.array([[Q + A * P[0, 0] * A, A * P[0, 0] * B], [B * P[0, 0] * A, B * P[0, 0] * B + R]])

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

    rms = {}
    final_states = {}
    for offset, alg in enumerate(("MC", "SARSA", "QLearning")):
        q_function = QuadraticQFunction(W0)
        rng = np.random.default_rng(config.seed + offset)
        rms_values = np.zeros(n_ep, dtype=float)
        last_states = None
        for episode in range(n_ep):
            learning_plant = NumericalPlant(
                LinearDiscreteDynamicsConfig([[A]], [[B]]),
                [x0],
                dt=dt,
                max_steps=kmax,
            )
            trajectory = run_value_episode(
                alg,
                learning_plant,
                q_function,
                rng,
                lambda state, action: float(
                    (Q * state[0] ** 2 + R * action[0] ** 2) / 2.0
                ),
                learning_rate=alpha,
                epsilon=config.exploration_rate,
                max_steps=kmax,
                stop=lambda state, step: np.linalg.norm(state) < tolerance,
            )
            last_states = trajectory.states
            rms_values[episode] = np.linalg.norm(q_function.weights - W_gt)
        key = alg.lower()
        rms[key] = rms_values
        final_states[key] = last_states
        write_csv(save_dir / f"{key}_final_states.csv", ("k", "x"), np.column_stack([np.arange(last_states.shape[0]), last_states[:, 0]]))

    write_csv(
        save_dir / "rms_errors.csv",
        ("episode", "sarsa", "qlearning", "mc"),
        np.column_stack([np.arange(n_ep), rms["sarsa"], rms["qlearning"], rms["mc"]]),
    )
    write_csv(save_dir / "lqr_states.csv", ("k", "x"), np.column_stack([np.arange(x_lqr.shape[0]), x_lqr[:, 0]]))
    summary = {
        "final_rms_mc": rms["mc"][-1],
        "final_rms_sarsa": rms["sarsa"][-1],
        "final_rms_qlearning": rms["qlearning"][-1],
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, rms, final_states, x_lqr)
    return summary


def _plot(save_dir, rms, final_states, x_lqr):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=False)
    for key, states in final_states.items():
        axes[0].plot(np.arange(states.shape[0]), states[:, 0], label=key)
    axes[0].plot(np.arange(x_lqr.shape[0]), x_lqr[:, 0], "--k", label="lqr")
    axes[0].set_title("Final Episode States")
    axes[1].plot(rms["sarsa"], label="sarsa")
    axes[1].plot(rms["qlearning"], label="qlearning")
    axes[1].plot(rms["mc"], label="mc")
    axes[1].set_title("Weight Error")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "value_learning.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
