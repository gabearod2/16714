from pathlib import Path
from typing import Optional
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.control.iterative_learning import TimeDomainILC
from spark_robot import AgiBotG1MobileBaseDynamic2Config

from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from .config import DEFAULT_CONFIG, Lecture13Config


DATA_OUTPUT_DIR = data_dir("lecture13")


def run(save_dir=DATA_OUTPUT_DIR, config: Lecture13Config = DEFAULT_CONFIG):
    save_dir = ensure_dir(save_dir)
    result = _run_time_domain_ilc(
        horizon=config.horizon,
        n_iter=config.iterations,
        dt=config.dt,
        sigma=config.disturbance_sigma,
        seed=config.seed,
        kp=config.kp,
        kd=config.kd,
        integrator=config.integrator,
    )
    write_json(save_dir / "config.json", asdict(config))
    t = result["t"]
    control_t = result["control_t"]
    states = result["states"]
    controls = result["controls"]
    final_idx = states.shape[0] - 1
    write_csv(save_dir / "states.csv", ("t", "x", "v", "reference"), np.column_stack([t, states[final_idx], result["reference"]]))
    write_csv(save_dir / "controls.csv", ("t", "u"), np.column_stack([control_t, controls[final_idx, :, 0]]))
    write_csv(save_dir / "error_norm.csv", ("iteration", "error_norm"), np.column_stack([np.arange(result["error_norm"].size), result["error_norm"]]))
    write_csv(save_dir / "learning_gain.csv", ("row",) + tuple(f"col{k}" for k in range(result["L"].shape[1])), np.column_stack([np.arange(result["L"].shape[0]), result["L"]]))
    write_json(save_dir / "summary.json", {"final_error_norm": result["error_norm"][-1], "initial_error_norm": result["error_norm"][0], "gain_shape": result["L"].shape})

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=False)
    axes[0].plot(t, result["reference"], "--", color="tab:gray", label="reference")
    for i in range(states.shape[0]):
        color = "k" if i == 0 else (i / max(1, states.shape[0] - 1), 0.0, 0.0)
        axes[0].plot(t, states[i, :, 0], color=color, linewidth=1.0)
    axes[0].set_title("Output")
    axes[1].plot(control_t, controls[-1, :, 0], color="tab:red")
    axes[1].set_title("Final Input")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "time_domain_ilc_trajectory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(np.arange(result["error_norm"].size), result["error_norm"], "o-")
    axes[0].set_title("ILC Error")
    axes[0].grid(True)
    image = axes[1].imshow(result["L"], aspect="auto")
    axes[1].set_title("Learning Gain")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / "time_domain_ilc_summary.png", dpi=160)
    plt.close(fig)
    return result


def _run_time_domain_ilc(
    horizon: int = 100,
    n_iter: int = 10,
    dt: float = 1.0,
    sigma: float = 0.0,
    seed: int = 714,
    reference: Optional[np.ndarray] = None,
    initial_response: Optional[np.ndarray] = None,
    scalar_A: Optional[float] = None,
    scalar_B: Optional[float] = None,
    kp: float = 0.1,
    kd: float = 0.5,
    integrator: str = "Euler",
):
    """Configure the Lecture 13 lifted-system ILC experiment."""
    rng = np.random.default_rng(seed)
    reference_function = lambda time: np.sin(0.2 * np.asarray(time, dtype=float))
    times = np.arange(horizon + 1, dtype=float) * dt
    reference_signal = (
        reference_function(times)
        if reference is None
        else np.asarray(reference, dtype=float).reshape(horizon + 1)
    )

    if scalar_A is None:
        dynamics_model = AgiBotG1MobileBaseDynamic2Config().create_dynamics_model(
            state_dof_names=["LinearX"], control_names=["aLinearX"]
        )
        K = np.array([[kp, kd]], dtype=float)
        C = np.array([[1.0, 0.0]], dtype=float)

        def feedback(state, time, _step):
            disturbance = sigma * rng.standard_normal() if sigma > 0.0 else 0.0
            return np.array(
                [
                    kp * (float(reference_function(time)) - state[0])
                    + kd
                    * (
                        (
                            float(reference_function(time + dt))
                            - float(reference_function(time))
                        )
                        / dt
                        - state[1]
                    )
                    + disturbance
                ],
                dtype=float,
            )

        def rollout(feedforward):
            def controller(state, time, step):
                return np.array(
                    [feedforward[step] + feedback(state, time, step)[0]],
                    dtype=float,
                )

            plant = NumericalPlant(
                dynamics_model,
                np.zeros(2),
                dt=dt,
                integrator=integrator,
                max_steps=horizon,
            )
            trajectory = rollout_model(
                plant,
                controller,
                max_steps=horizon,
            )
            return trajectory.times, trajectory.states, trajectory.controls

        _, feedback_states, feedback_controls = rollout(
            np.zeros(horizon, dtype=float)
        )
        output = feedback_states[:, 0]
    else:
        if scalar_B is None or initial_response is None:
            raise ValueError(
                "scalar_B and initial_response are required when scalar_A is provided."
            )
        A = np.array([[scalar_A]], dtype=float)
        B = np.array([[scalar_B]], dtype=float)
        K = np.array([[0.0]], dtype=float)
        C = np.array([[1.0]], dtype=float)
        feedback_states = np.asarray(initial_response, dtype=float).reshape(
            horizon + 1, 1
        )
        feedback_controls = np.zeros((horizon, 1), dtype=float)
        output = feedback_states[:, 0]
        rollout = None

    if scalar_A is None:
        controller = TimeDomainILC.from_dynamics_model(
            dynamics_model,
            K=K,
            C=C,
            horizon=horizon,
            dt=dt,
            discretization=integrator,
        )
    else:
        controller = TimeDomainILC(A, B, K, C, horizon)
    errors = np.zeros((n_iter, horizon + 1), dtype=float)
    feedforward = np.zeros((n_iter, horizon), dtype=float)
    norms = np.zeros(n_iter, dtype=float)
    states = [feedback_states]
    controls = [feedback_controls]
    for iteration in range(n_iter):
        errors[iteration] = reference_signal - output
        previous = (
            feedforward[iteration - 1]
            if iteration > 0
            else np.zeros(horizon)
        )
        feedforward[iteration] = controller.update_feedforward(
            errors[iteration], previous
        )
        if rollout is not None:
            _, feedback_states, feedback_controls = rollout(feedforward[iteration])
            output = feedback_states[:, 0]
        states.append(feedback_states.copy())
        controls.append(feedback_controls.copy())
        norms[iteration] = np.linalg.norm(errors[iteration])
    return {
        "t": times,
        "control_t": times[:-1],
        "reference": reference_signal,
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "errors": errors,
        "feedforward": feedforward,
        "error_norm": norms,
        "L": controller.L,
    }


if __name__ == "__main__":
    run()
