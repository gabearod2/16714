"""Small policy parameterizations used by the RL lectures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearGaussianActor:
    mu: float
    sigma: float
    mu_bounds: tuple[float, float] | None = None
    sigma_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.mu = float(self.mu)
        self.sigma = float(self.sigma)
        self.project()

    def sample(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        return np.array([self.mu * x + rng.standard_normal() * self.sigma * x])

    def grad_log_prob(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        u = float(np.asarray(action, dtype=float).reshape(-1)[0])
        denominator = (self.sigma * x) ** 2
        if abs(denominator) < 1.0e-14:
            return np.zeros(2, dtype=float)
        return np.array(
            [
                (u - self.mu * x) / denominator * x,
                -x + (u - self.mu * x) ** 2 / denominator * x,
            ],
            dtype=float,
        )

    @property
    def parameters(self) -> np.ndarray:
        return np.array([self.mu, self.sigma], dtype=float)

    def apply_cost_gradient(self, gradient: np.ndarray) -> None:
        gradient = np.asarray(gradient, dtype=float).reshape(2)
        self.mu -= float(gradient[0])
        self.sigma -= float(gradient[1])
        self.project()

    def project(self) -> None:
        self.mu = _project_scalar(self.mu, self.mu_bounds, default=0.0)
        self.sigma = _project_scalar(self.sigma, self.sigma_bounds, default=1.0e-6)


def bounded_step(scale: float, direction: np.ndarray, limit: float = 1.0e6) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        step = float(scale) * np.asarray(direction, dtype=float)
    return np.clip(
        np.nan_to_num(step, nan=0.0, posinf=limit, neginf=-limit),
        -limit,
        limit,
    )


def _project_scalar(
    value: float,
    bounds: tuple[float, float] | None,
    *,
    default: float,
) -> float:
    if bounds is None:
        return float(np.nan_to_num(value, nan=default, posinf=1.0e6, neginf=-1.0e6))
    low, high = bounds
    return float(
        np.clip(
            np.nan_to_num(value, nan=default, posinf=high, neginf=low),
            low,
            high,
        )
    )

