from pathlib import Path
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.estimation import GradientParameterEstimator, RecursiveLeastSquaresEstimator
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture17Config


DATA_OUTPUT_DIR = data_dir("lecture17")


def run(save_dir=DATA_OUTPUT_DIR, config: Lecture17Config = DEFAULT_CONFIG):
    save_dir = ensure_dir(save_dir)
    N = config.horizon
    dt = config.dt
    x = np.zeros(N + 1, dtype=float)
    p_true = np.zeros(N, dtype=float)
    x[0] = config.initial_state
    for k in range(N):
        t = k * dt
        p_true[k] = config.nominal_parameter + config.variation * np.sin(t / 10.0)
        x[k + 1] = p_true[k] * np.sin(x[k])

    rls = RecursiveLeastSquaresEstimator(
        p0hat=config.nominal_parameter,
        H0=config.rls_information,
        lam=config.forgetting_factor,
    )
    sgd = GradientParameterEstimator(
        p0hat=config.nominal_parameter, H0=config.rls_information
    )
    rls_p = np.zeros(N + 1, dtype=float)
    sgd_p = np.zeros(N + 1, dtype=float)
    rls_H = np.zeros(N + 1, dtype=float)
    rls_p[0] = rls.phat
    sgd_p[0] = sgd.phat
    rls_H[0] = rls.H
    for k in range(N):
        rls_p[k + 1], rls_H[k + 1] = rls.step(x[k], x[k + 1])
        sgd_p[k + 1] = sgd.step(x[k], x[k + 1])

    t = np.arange(N + 1) * dt
    write_csv(save_dir / "states.csv", ("t", "x"), np.column_stack([t, x]))
    write_csv(
        save_dir / "parameter_estimates.csv",
        ("t", "p_true", "rls", "sgd", "rls_learning_rate", "sgd_learning_rate"),
        np.column_stack([t, np.r_[p_true, np.nan], rls_p, sgd_p, 1.0 / rls_H, np.full(N + 1, 1.0 / config.rls_information)]),
    )
    summary = {
        "rls_final_error": abs(rls_p[-1] - p_true[-1]),
        "sgd_final_error": abs(sgd_p[-1] - p_true[-1]),
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=False)
    axes[0].plot(t, x, "k", label="x")
    axes[0].set_title("State")
    axes[1].plot(t[:-1], p_true, "k", label="true")
    axes[1].plot(t, rls_p, "r", label="rls")
    axes[1].plot(t, sgd_p, "b", label="sgd")
    axes[1].set_title("Parameter")
    axes[2].plot(t, 1.0 / rls_H, "r", label="rls")
    axes[2].plot(t, np.full(N + 1, 1.0 / config.rls_information), "--b", label="sgd")
    axes[2].set_title("Learning Rate")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "rls_sgd.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    run()
