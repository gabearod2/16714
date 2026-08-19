from pathlib import Path
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from spark_policy.control.iterative_learning import TimeDomainILC
from spark_policy.utils.control_math import infinite_lqr
from spark_robot import DiscreteTimeDynamicsConfig, LinearDiscreteDynamicsConfig
from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from shared.numerical_rollout import NumericalPlant, rollout_model
from shared.rl import (
    LinearGaussianActor,
    ScalarQuadraticCritic,
    run_actor_critic,
    run_reinforce,
)
from .config import DEFAULT_CONFIG, Lecture22Config


DATA_OUTPUT_DIR = data_dir("lecture22")


def run(
    save_dir=DATA_OUTPUT_DIR,
    seed: int | None = None,
    scenario: str | None = None,
    config: Lecture22Config = DEFAULT_CONFIG,
):
    if seed is not None:
        config = replace(config, seed=int(seed))
    if scenario is not None:
        config = replace(config, scenario=str(scenario))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)
    A, B, Q, R = config.A, config.B, config.Q, config.R
    x0, kmax, dt = config.initial_state, config.horizon, config.dt
    n_ep, scenario = config.episodes, config.scenario
    seeds = rng.integers(1, 1_000_000, size=n_ep)
    K, _P = infinite_lqr(np.array([[A]]), np.array([[B]]), np.array([[Q]]), np.array([[R]]))

    def stage_cost(x, u):
        return float((Q * x[0] ** 2 + R * u[0] ** 2) / 2.0)

    def nominal_f(x, u, _t, _k):
        return np.array([A * x[0] + B * u[0]], dtype=float)

    nominal_plant = NumericalPlant(
        LinearDiscreteDynamicsConfig([[A]], [[B]]), [x0], dt=dt, max_steps=kmax
    )
    x_lqr = rollout_model(
        nominal_plant,
        lambda state, _time, _step: -K @ state,
        max_steps=kmax,
    ).states
    first_rng = np.random.default_rng(int(seeds[0]))
    x_fb = _run_transition(
        _scenario_dynamics(A, B, scenario, first_rng),
        lambda x, _t, _k: -K @ x,
        x0,
        kmax,
        dt,
    ).states

    pg_results = {}
    for offset, alg in enumerate(("REINFORCE", "REINFORCEbl", "ActorCritic")):
        actor = LinearGaussianActor(
            mu=config.initial_mu,
            sigma=config.initial_sigma,
            mu_bounds=config.mu_bounds,
            sigma_bounds=config.sigma_bounds,
        )
        critic = ScalarQuadraticCritic(config.initial_value_weight)
        rms = np.zeros(n_ep, dtype=float)
        std = np.zeros(n_ep, dtype=float)
        last_states = None
        for episode in range(n_ep):
            ep_rng = np.random.default_rng(int(seeds[episode]))
            f_dt = _scenario_dynamics(A, B, scenario, ep_rng)
            learning_plant = _make_transition_plant(f_dt, x0, kmax, dt)
            stop = lambda state, step: np.linalg.norm(state) < config.tolerance
            if alg == "ActorCritic":
                trajectory = run_actor_critic(
                    learning_plant,
                    actor,
                    critic,
                    ep_rng,
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
                    ep_rng,
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
        pg_results[key] = {"rms": rms, "std": std, "states": last_states}

    ilc = _run_ilc_comparison(A, B, x0, kmax, dt, n_ep, seeds, scenario, x_lqr[:, 0], x_fb[:, 0])

    write_csv(
        save_dir / "policy_gradient_metrics.csv",
        ("episode", "reinforce", "reinforce_bl", "actor_critic", "reinforce_std", "reinforce_bl_std", "actor_critic_std"),
        np.column_stack(
            [
                np.arange(n_ep),
                pg_results["reinforce"]["rms"],
                pg_results["reinforcebl"]["rms"],
                pg_results["actorcritic"]["rms"],
                pg_results["reinforce"]["std"],
                pg_results["reinforcebl"]["std"],
                pg_results["actorcritic"]["std"],
            ]
        ),
    )
    write_csv(save_dir / "ilc_error_norm.csv", ("iteration", "error_norm"), np.column_stack([np.arange(n_ep), ilc["error_norm"]]))
    write_csv(save_dir / "lqr_states.csv", ("k", "x"), np.column_stack([np.arange(x_lqr.shape[0]), x_lqr[:, 0]]))
    write_csv(save_dir / "ilc_final_states.csv", ("k", "x"), np.column_stack([np.arange(ilc["states"][-1].shape[0]), ilc["states"][-1][:, 0]]))
    summary = {
        "scenario": scenario,
        "final_rms_reinforce": pg_results["reinforce"]["rms"][-1],
        "final_rms_reinforce_bl": pg_results["reinforcebl"]["rms"][-1],
        "final_rms_actor_critic": pg_results["actorcritic"]["rms"][-1],
        "final_ilc_error": ilc["error_norm"][-1],
        "seed": config.seed,
    }
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, pg_results, ilc, x_lqr)
    return summary


def _scenario_dynamics(A, B, scenario, rng):
    def f_dt(x, u, t, _k):
        base = A * x[0] + B * u[0]
        if scenario == "repetitive":
            disturbance = np.sin(0.1 * np.pi * t)
        elif scenario == "state-dependent":
            disturbance = np.sin(0.1 * np.pi * t) * x[0]
        else:
            disturbance = rng.standard_normal()
        return np.array([base + disturbance], dtype=float)

    return f_dt


def _run_ilc_comparison(A, B, x0, kmax, dt, n_ep, seeds, scenario, reference, initial_response):
    nominal_model = LinearDiscreteDynamicsConfig([[A]], [[B]]).create_dynamics_model()
    policy = TimeDomainILC.from_dynamics_model(
        nominal_model,
        K=np.array([[0.0]]),
        C=np.array([[1.0]]),
        horizon=kmax,
        dt=dt,
        discretization="native",
    )
    ff = np.zeros((n_ep, kmax), dtype=float)
    errors = np.zeros((n_ep, kmax + 1), dtype=float)
    norms = np.zeros(n_ep, dtype=float)
    states = [initial_response.reshape(-1, 1)]
    x_fb = initial_response.copy()
    for i in range(n_ep):
        errors[i] = reference[: kmax + 1] - x_fb[: kmax + 1]
        ff[i] = policy.update_feedforward(
            errors[i], ff[i - 1] if i > 0 else np.zeros(kmax)
        )
        ep_rng = np.random.default_rng(int(seeds[i]))
        f_dt = _scenario_dynamics(A, B, scenario, ep_rng)
        x_new = _run_transition(
            f_dt,
            lambda _x, _t, k: np.array([ff[i, k]], dtype=float),
            x0,
            kmax,
            dt,
        ).states
        x_fb = x_new[:, 0]
        states.append(x_new)
        norms[i] = np.linalg.norm(errors[i])
    return {"errors": errors, "error_norm": norms, "feedforward": ff, "states": states}


def _run_transition(transition, controller, initial_state, steps, dt):
    plant = _make_transition_plant(transition, initial_state, steps, dt)
    return rollout_model(
        plant,
        controller,
        max_steps=steps,
    )


def _make_transition_plant(transition, initial_state, steps, dt):
    robot_cfg = DiscreteTimeDynamicsConfig(
        1,
        1,
        lambda state, control, context, _parameters: transition(
            state, control, context.time, context.step_index
        ),
        dynamics_variant="lecture22_disturbed_discrete_system",
    )
    return NumericalPlant(
        robot_cfg, [initial_state], dt=dt, max_steps=steps
    )


def _plot(save_dir, pg_results, ilc, x_lqr):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=False)
    axes[0].plot(pg_results["reinforce"]["rms"], "b", label="reinforce")
    axes[0].plot(pg_results["reinforcebl"]["rms"], "r", label="reinforce baseline")
    axes[0].plot(pg_results["actorcritic"]["rms"], "k", label="actor critic")
    axes[0].set_title("Policy Parameter Error")
    axes[1].plot(ilc["error_norm"], "o-", label="ilc")
    axes[1].set_title("ILC Error")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "pg_ilc_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x_lqr[:, 0], "--k", label="lqr")
    ax.plot(ilc["states"][-1][:, 0], "r", label="ilc final")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("ILC Final State")
    fig.tight_layout()
    fig.savefig(save_dir / "ilc_final_state.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
