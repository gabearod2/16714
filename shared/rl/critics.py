"""Value-function parameterizations used by course experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalarQuadraticCritic:
    weight: float

    def value(self, state: np.ndarray) -> float:
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        return float(self.weight * x**2 / 2.0)

    @staticmethod
    def gradient(state: np.ndarray) -> float:
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        return float(x**2 / 2.0)

