"""Known-plant convergence analysis for the Lecture 19 MRAC experiment."""

import numpy as np


def analyze_mrac_history(controller, true_A, true_B):
    true_A = np.asarray(true_A, dtype=float)
    true_B = np.asarray(true_B, dtype=float)
    history_length = len(controller.F_history)
    values = np.zeros(history_length, dtype=float)
    posterior = np.zeros((history_length, controller.n), dtype=float)
    regressors = [None] + list(controller.phi_history)
    for index in range(history_length):
        parameter_error = np.hstack(
            [
                controller.Ahat_history[index] - true_A,
                controller.Bhat_history[index] - true_B,
            ]
        )
        if index > 0 and regressors[index] is not None:
            regressor = regressors[index]
            posterior[index] = (
                1.0 - regressor.T @ controller.F_history[index] @ regressor
            ) * controller.error_history[index]
        else:
            posterior[index] = controller.error_history[index]
        values[index] = np.linalg.norm(posterior[index]) ** 2 + np.trace(
            parameter_error
            @ np.linalg.inv(controller.F_history[index])
            @ parameter_error.T
        )
    return values, posterior

