from pathlib import Path
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.control.iterative_learning import FrequencyDomainILC
from spark_robot import AgiBotG1MobileBaseDynamic2Config

from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from .config import DEFAULT_CONFIG, Lecture12Config


DATA_OUTPUT_DIR = data_dir("lecture12")


def run(save_dir=DATA_OUTPUT_DIR, config: Lecture12Config = DEFAULT_CONFIG):
    save_dir = ensure_dir(save_dir)
    result = _run_frequency_ilc(
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
    _save_ilc_outputs(save_dir, result, "frequency_ilc")
    return result


def _run_frequency_ilc(
    horizon: int = 100,
    n_iter: int = 10,
    dt: float = 1.0,
    sigma: float = 0.0,
    seed: int = 714,
    kp: float = 0.1,
    kd: float = 0.5,
    integrator: str = "Euler",
):
    """Configure the Lecture 12 plant, reference, and feedback experiment."""
    rng = np.random.default_rng(seed)
    reference = lambda time: np.sin(0.2 * np.asarray(time, dtype=float))
    times = np.arange(horizon + 1, dtype=float) * dt
    dynamics_model = AgiBotG1MobileBaseDynamic2Config().create_dynamics_model(
        state_dof_names=["LinearX"], control_names=["aLinearX"]
    )

    def feedback(state, time, _step):
        disturbance = sigma * rng.standard_normal() if sigma > 0.0 else 0.0
        return np.array(
            [
                kp * (float(reference(time)) - state[0])
                + kd
                * (
                    (float(reference(time + dt)) - float(reference(time))) / dt
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
        np.zeros(horizon + 1, dtype=float)
    )
    controller = FrequencyDomainILC(
        a=np.array([1.0, -2.0, 1.0]),
        b=np.array([dt]),
        c=np.array([-kd / dt, kd / dt - kp]),
    )
    errors = np.zeros((n_iter, horizon + 1), dtype=float)
    feedforward = np.zeros((n_iter, horizon + 1), dtype=float)
    norms = np.zeros(n_iter, dtype=float)
    states = [feedback_states]
    controls = [feedback_controls]
    for iteration in range(n_iter):
        errors[iteration] = reference(times) - feedback_states[:, 0]
        previous = (
            feedforward[iteration - 1]
            if iteration > 0
            else np.zeros(horizon + 1)
        )
        feedforward[iteration] = controller.update_feedforward(
            errors[iteration], previous
        )
        _, feedback_states, feedback_controls = rollout(feedforward[iteration])
        states.append(feedback_states)
        controls.append(feedback_controls)
        norms[iteration] = np.linalg.norm(errors[iteration])
    return {
        "t": times,
        "control_t": times[:-1],
        "reference": reference(times),
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "errors": errors,
        "feedforward": feedforward,
        "error_norm": norms,
        "L_b": controller.L_b,
        "L_a": controller.L_a,
    }


def _save_ilc_outputs(save_dir, result, prefix):
    t = result["t"]
    control_t = result["control_t"]
    states = result["states"]
    controls = result["controls"]
    final_idx = states.shape[0] - 1
    write_csv(save_dir / "states.csv", ("t", "x", "v", "reference"), np.column_stack([t, states[final_idx], result["reference"]]))
    write_csv(save_dir / "controls.csv", ("t", "u"), np.column_stack([control_t, controls[final_idx, :, 0]]))
    write_csv(save_dir / "error_norm.csv", ("iteration", "error_norm"), np.column_stack([np.arange(result["error_norm"].size), result["error_norm"]]))
    write_csv(save_dir / "feedforward.csv", ("iteration",) + tuple(f"u{k}" for k in range(result["feedforward"].shape[1])), np.column_stack([np.arange(result["feedforward"].shape[0]), result["feedforward"]]))
    write_json(save_dir / "summary.json", {"final_error_norm": result["error_norm"][-1], "initial_error_norm": result["error_norm"][0], "L_b": result["L_b"], "L_a": result["L_a"]})

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=False)
    axes[0].plot(t, result["reference"], "--", color="tab:gray", label="reference")
    for i in range(states.shape[0]):
        color = "k" if i == 0 else (i / max(1, states.shape[0] - 1), 0.0, 0.0)
        axes[0].plot(t, states[i, :, 0], color=color, linewidth=1.0)
    axes[0].set_title("Output")
    axes[1].plot(control_t, controls[-1, :, 0], color="tab:red", label="final")
    axes[1].set_title("Final Input")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}_trajectory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(np.arange(result["error_norm"].size), result["error_norm"], "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error norm")
    ax.set_title("ILC Error")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}_error.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
