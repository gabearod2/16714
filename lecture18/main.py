from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

from spark_policy.utils.control_math import infinite_lqr
from spark_robot import AgiBotG1MobileBaseDynamic2Config
from spark_policy.estimation import SteadyStateKalmanFilterEstimator
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture18Config


DATA_OUTPUT_DIR = data_dir("lecture18")


def run(save_dir=DATA_OUTPUT_DIR, seed=None, config: Lecture18Config = DEFAULT_CONFIG):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)
    dim = 4
    dt = config.dt
    N = config.horizon
    dynamics_model = AgiBotG1MobileBaseDynamic2Config().create_dynamics_model(
        state_dof_names=["LinearX", "LinearY"],
        control_names=["aLinearX", "aLinearY"],
    )
    A, B = dynamics_model.discrete_matrices(dt, "ZOH")
    C = np.hstack([np.eye(2), np.zeros((2, 2))])
    W = config.process_variance * np.eye(2)
    V = config.measurement_variance * np.eye(2)
    Bw = B.copy()
    x0 = np.array([10.0, 5.0, 5.0, 0.0])
    X0 = 0.001 * np.eye(dim)
    x0hat = x0 + np.sqrt(X0) @ rng.standard_normal(dim)
    Q = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), np.zeros((2, 2))]])
    R = np.eye(2)
    K, _P = infinite_lqr(A, B, Q, R)
    est = SteadyStateKalmanFilterEstimator(A, B, C, Bw, W, V, x0hat, X0)

    states = np.zeros((N + 1, dim), dtype=float)
    xhat = np.zeros((N + 1, dim), dtype=float)
    controls = np.zeros((N, 2), dtype=float)
    measurements = np.zeros((N + 1, 2), dtype=float)
    states[0] = x0
    xhat[0] = x0hat
    measurements[0] = C @ states[0] + sqrtm(V).real @ rng.standard_normal(2)
    for k in range(N):
        controls[k] = -K @ xhat[k]
        states[k + 1] = A @ states[k] + B @ controls[k] + Bw @ (sqrtm(W).real @ rng.standard_normal(2))
        measurements[k + 1] = C @ states[k + 1] + sqrtm(V).real @ rng.standard_normal(2)
        xhat[k + 1], _ = est.step(controls[k], measurements[k + 1])

    t = np.arange(N + 1) * dt
    write_csv(save_dir / "states.csv", ("t", "x1", "x2", "v1", "v2"), np.column_stack([t, states]))
    write_csv(save_dir / "controls.csv", ("t", "u1", "u2"), np.column_stack([t[:-1], controls]))
    write_csv(save_dir / "measurements.csv", ("t", "y1", "y2"), np.column_stack([t, measurements]))
    write_csv(save_dir / "estimates.csv", ("t", "xhat1", "xhat2", "vhat1", "vhat2"), np.column_stack([t, xhat]))
    summary = {
        "state_estimate_error": np.linalg.norm(xhat - states),
        "final_state_norm": np.linalg.norm(states[-1]),
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, states, measurements, xhat)
    return summary


def _plot(save_dir, states, measurements, xhat):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].plot(states[:, 0], states[:, 1], "k", label="state")
    axes[0].plot(measurements[:, 0], measurements[:, 1], "r.", label="measurement")
    axes[0].plot(xhat[:, 0], xhat[:, 1], "b", label="estimate")
    axes[0].set_title("Position Phase Plot")
    for i in range(4):
        axes[1].plot(states[:, i], "k", alpha=0.45, label="state" if i == 0 else None)
        axes[1].plot(xhat[:, i], "b", alpha=0.7, label="estimate" if i == 0 else None)
    axes[1].set_title("States And Estimates")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "separation_principle.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
