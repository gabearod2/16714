"""Course-only reinforcement-learning data types."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Episode:
    times: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    costs: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

