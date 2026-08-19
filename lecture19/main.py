from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.control.adaptive import ModelReferenceAdaptiveController
from spark_robot import LinearDiscreteDynamicsConfig
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from .analysis import analyze_mrac_history
from .config import DEFAULT_CONFIG, Lecture19Config


DATA_OUTPUT_DIR = data_dir("lecture19")


def run(
    save_dir=DATA_OUTPUT_DIR,
    seed: int | None = None,
    config: Lecture19Config = DEFAULT_CONFIG,
):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)
    n = config.state_dim
    m = config.control_dim
    A = rng.random((n, n))
    B = rng.random((n, m))
    Astar = np.zeros((n, n), dtype=float)
    Ahat = np.eye(n)
    Bhat = np.eye(n, m)
    x0 = np.asarray(config.initial_state, dtype=float)
    F = 10.0 * np.eye(n + m) + np.ones((n + m, n + m))
    N = config.horizon
    dt = config.dt
    policy = ModelReferenceAdaptiveController(Astar, Ahat, Bhat, F, x0=x0)
    reference = np.zeros((N + 1, n), dtype=float)
    reference[0] = x0
    for step in range(N):
        reference[step + 1] = Astar @ reference[step]

    plant = NumericalPlant(
        LinearDiscreteDynamicsConfig(A, B), x0, dt=dt, max_steps=N
    )
    trajectory = rollout_model(
        plant,
        lambda state, _time, step: policy.compute_control(state, reference[step]),
        max_steps=N,
    )
    t, states, controls = trajectory.times, trajectory.states, trajectory.controls
    values, posterior = analyze_mrac_history(policy, A, B)
    hist_t = np.arange(values.size, dtype=float) * dt

    write_csv(save_dir / "states.csv", ("t", "x1", "x2"), np.column_stack([t, states]))
    write_csv(save_dir / "controls.csv", ("t", "u1", "u2"), np.column_stack([t[:-1], controls]))
    write_csv(save_dir / "analysis.csv", ("t", "value", "posterior_error_1", "posterior_error_2"), np.column_stack([hist_t, values, posterior]))
    write_json(
        save_dir / "summary.json",
        {
            "final_state_norm": np.linalg.norm(states[-1]),
            "final_value": values[-1],
            "seed": config.seed,
            "A": A,
            "B": B,
        },
    )
    write_json(save_dir / "config.json", asdict(config))
    _plot(save_dir, states, values, posterior)
    return {"final_state_norm": np.linalg.norm(states[-1]), "final_value": values[-1]}


def _plot(save_dir, states, values, posterior):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(states[:, 0], "k", label="x1")
    ax.plot(states[:, 1], color="0.35", label="x2")
    ax.plot(values, "r", label="value")
    ax.plot(posterior[:, 0], "-*b", label="posterior error 1")
    ax.plot(posterior[:, 1], "-*c", label="posterior error 2")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Model Reference Adaptive Control")
    fig.tight_layout()
    fig.savefig(save_dir / "mrac.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
