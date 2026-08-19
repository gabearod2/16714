from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.estimation import ExtendedKalmanFilterEstimator, UnscentedKalmanFilterEstimator
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture16Config


DATA_OUTPUT_DIR = data_dir("lecture16")


def run(save_dir=DATA_OUTPUT_DIR, seed=None, config: Lecture16Config = DEFAULT_CONFIG):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)

    x0 = np.array([4.0, 1.0])
    X0 = 0.01 * np.array([[8.0, 2.0], [2.0, 3.0]])
    x0hat = x0 + np.sqrt(X0) @ rng.standard_normal(2)
    W = np.ones((2, 2)) * config.process_variance
    V = np.ones((2, 2)) * config.measurement_variance
    Bw = np.ones((2, 1))
    N = config.horizon
    dt = config.dt

    def f_nominal(x, u):
        return np.asarray(x, dtype=float).reshape(2) + np.asarray(u, dtype=float).reshape(2)

    def h_nominal(x):
        x = np.asarray(x, dtype=float).reshape(2)
        return np.array([x[0] ** 4 * x[1], x[0] + x[1] ** 5], dtype=float)

    def A_func(_x, _u):
        return np.eye(2)

    def C_func(x):
        x = np.asarray(x, dtype=float).reshape(2)
        return np.array([[4.0 * x[0] ** 3 * x[1], x[0] ** 4], [1.0, 5.0 * x[1] ** 4]], dtype=float)

    states = np.zeros((N + 1, 2), dtype=float)
    controls = np.zeros((N, 2), dtype=float)
    measurements = np.zeros((N + 1, 2), dtype=float)
    states[0] = x0
    for k in range(N):
        controls[k] = 0.0
        measurements[k] = h_nominal(states[k]) + np.sqrt(config.measurement_variance) * rng.standard_normal()
        states[k + 1] = f_nominal(states[k], controls[k]) + np.ones(2) * np.sqrt(config.process_variance) * rng.standard_normal()
    measurements[N] = h_nominal(states[N]) + np.sqrt(config.measurement_variance) * rng.standard_normal()

    process_covariance = np.array([[config.process_variance]])
    ekf = ExtendedKalmanFilterEstimator(f_nominal, h_nominal, A_func, C_func, Bw, process_covariance, V, x0hat, X0)
    ukf = UnscentedKalmanFilterEstimator(f_nominal, h_nominal, Bw, process_covariance, V, x0hat, X0)
    ekf_x = np.zeros((N, 2), dtype=float)
    ukf_x = np.zeros((N, 2), dtype=float)
    ekf_z = np.zeros((N, 2, 2), dtype=float)
    ukf_z = np.zeros((N, 2, 2), dtype=float)
    ekf_x[0] = x0hat
    ukf_x[0] = x0hat
    ekf_z[0] = X0
    ukf_z[0] = X0
    for k in range(N - 1):
        ekf_x[k + 1], ekf_z[k + 1] = ekf.step(controls[k], measurements[k + 1])
        ukf_x[k + 1], ukf_z[k + 1] = ukf.step(controls[k], measurements[k + 1])

    t = np.arange(N + 1) * dt
    write_csv(save_dir / "states.csv", ("t", "x1", "x2"), np.column_stack([t, states]))
    write_csv(save_dir / "measurements.csv", ("t", "y1", "y2"), np.column_stack([t, measurements]))
    write_csv(save_dir / "ekf_estimates.csv", ("t", "xhat1", "xhat2"), np.column_stack([t[:N], ekf_x]))
    write_csv(save_dir / "ukf_estimates.csv", ("t", "xhat1", "xhat2"), np.column_stack([t[:N], ukf_x]))
    write_csv(save_dir / "ekf_covariance.csv", ("t", "z11", "z12", "z21", "z22"), np.column_stack([t[:N], ekf_z.reshape(N, 4)]))
    write_csv(save_dir / "ukf_covariance.csv", ("t", "z11", "z12", "z21", "z22"), np.column_stack([t[:N], ukf_z.reshape(N, 4)]))
    summary = {
        "ekf_error": np.linalg.norm(ekf_x - states[:N]),
        "ukf_error": np.linalg.norm(ukf_x - states[:N]),
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, t, states, ekf_x, ukf_x, ekf_z, ukf_z)
    return summary


def _plot(save_dir, t, states, ekf_x, ukf_x, ekf_z, ukf_z):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=False)
    for idx, ax in enumerate(axes[:2]):
        ax.plot(t, states[:, idx], "k", label=f"x{idx + 1}")
        ax.plot(t[: ekf_x.shape[0]], ekf_x[:, idx], "--b", label="ekf")
        ax.plot(t[: ukf_x.shape[0]], ukf_x[:, idx], "b", label="ukf")
        ax.grid(True)
        ax.legend(loc="best")
    axes[2].plot(t[: ekf_z.shape[0]], ekf_z[:, 0, 0], "--b", label="ekf z11")
    axes[2].plot(t[: ukf_z.shape[0]], ukf_z[:, 0, 0], "b", label="ukf z11")
    axes[2].set_title("Covariance Entry")
    axes[2].grid(True)
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "ekf_ukf.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
