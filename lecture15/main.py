from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.estimation import KalmanFilterEstimator, SteadyStateKalmanFilterEstimator
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture15Config


DATA_OUTPUT_DIR = data_dir("lecture15")


def run(save_dir=DATA_OUTPUT_DIR, seed=None, config: Lecture15Config = DEFAULT_CONFIG):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)

    A = np.array([[0.5]])
    B = np.array([[1.0]])
    Bw = np.array([[1.0]])
    C = np.array([[1.0]])
    W = np.array([[config.process_variance]])
    V = np.array([[config.measurement_variance]])
    N = config.horizon
    dt = config.dt
    x0 = np.array([0.0])
    X0 = np.array([[1.0]])
    x0hat = x0 + rng.standard_normal(1) * np.sqrt(X0[0, 0])

    states = np.zeros((N + 1, 1), dtype=float)
    controls = np.zeros((N, 1), dtype=float)
    measurements = np.zeros((N, 1), dtype=float)
    states[0] = x0
    for k in range(N):
        t = k * dt
        controls[k, 0] = np.sin(t / (N / 2.0 / np.pi))
        measurements[k] = C @ states[k] + np.sqrt(V[0, 0]) * rng.standard_normal(1)
        states[k + 1] = A @ states[k] + B @ controls[k] + Bw @ (np.sqrt(W[0, 0]) * rng.standard_normal(1))

    kf = KalmanFilterEstimator(A, B, C, Bw, W, V, x0hat, X0)
    kfss = SteadyStateKalmanFilterEstimator(A, B, C, Bw, W, V, x0hat, X0)
    xhat = np.zeros((N, 1), dtype=float)
    xhat_ss = np.zeros((N, 1), dtype=float)
    z = np.zeros(N, dtype=float)
    z_ss = np.zeros(N, dtype=float)
    xhat[0] = x0hat
    xhat_ss[0] = x0hat
    z[0] = X0[0, 0]
    z_ss[0] = X0[0, 0]
    for k in range(N - 1):
        xhat[k + 1], zk = kf.step(controls[k], measurements[k + 1])
        xhat_ss[k + 1], zsk = kfss.step(controls[k], measurements[k + 1])
        z[k + 1] = zk[0, 0]
        z_ss[k + 1] = zsk[0, 0]

    t = np.arange(N + 1) * dt
    write_csv(save_dir / "states.csv", ("t", "x"), np.column_stack([t, states]))
    write_csv(save_dir / "controls.csv", ("t", "u"), np.column_stack([t[:-1], controls]))
    write_csv(
        save_dir / "estimates.csv",
        ("t", "x", "y", "kf", "kfss", "kf_var", "kfss_var"),
        np.column_stack([t[:N], states[:N, 0], measurements[:, 0], xhat[:, 0], xhat_ss[:, 0], z, z_ss]),
    )
    summary = {
        "measurement_error": np.linalg.norm(measurements[:, 0] - states[:N, 0]),
        "kf_error": np.linalg.norm(xhat[:, 0] - states[:N, 0]),
        "kfss_error": np.linalg.norm(xhat_ss[:, 0] - states[:N, 0]),
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, t, states, measurements, xhat, xhat_ss, z, z_ss)
    return summary


def _plot(save_dir, t, states, measurements, xhat, xhat_ss, z, z_ss):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=False)
    axes[0].plot(t, states[:, 0], "k", label="state")
    axes[0].plot(t[: measurements.shape[0]], measurements[:, 0], "r", label="measurement")
    axes[0].plot(t[: xhat.shape[0]], xhat[:, 0], "b", label="kf")
    axes[0].plot(t[: xhat_ss.shape[0]], xhat_ss[:, 0], "--b", label="kfss")
    axes[0].set_title("Kalman Estimates")
    axes[1].plot(t[: z.size], z, label="kf")
    axes[1].plot(t[: z_ss.size], z_ss, label="kfss")
    axes[1].set_title("Posterior Variance")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "kalman_filter.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
