"""Provided Euler transitions and parameters for HW1."""

from dataclasses import asdict

import numpy as np
from spark_robot import BicycleParams

from .unicycle_bicycle_helpers import wrap_angle


DEFAULT_BICYCLE_PARAMS = BicycleParams()


def unicycle_transition(state, control, _context, parameters):
    # x = [ p_x   ]    u = [ v_dot     ]
    #     [ p_y   ]        [ theta_dot ]
    #     [ v     ]
    #     [ theta ]
    # x_dot = [ v*cos(theta) ] + [ 0  0 ] u;   x_next = x + dt*x_dot
    #         [ v*sin(theta) ]   [ 0  0 ]
    #         [ 0            ]   [ 1  0 ]
    #         [ 0            ]   [ 0  1 ]
    p_x, p_y, velocity, heading = state
    acceleration, turning_rate = control
    dt = parameters["dt"]
    return np.array(
        [
            p_x + dt * velocity * np.cos(heading),
            p_y + dt * velocity * np.sin(heading),
            velocity + dt * acceleration,
            wrap_angle(heading + dt * turning_rate),
        ]
    )


def bicycle_transition(state, control, _context, parameters):
    # x_b = [X, Y, v_x, v_y, r, psi]^T, u_b = [a_x, delta]^T.
    _, _, v_x, v_y, yaw_rate, heading = state
    acceleration, steering = control
    p, dt = parameters, parameters["dt"]
    steering = np.clip(steering, -p["steering_limit"], p["steering_limit"])
    # Linear tire slip angles use a calibrated forward speed near rest.
    v_x_effective = np.copysign(
        max(abs(v_x), p["minimum_forward_speed"]), v_x if v_x else 1.0
    )
    alpha_front = (
        np.arctan2(v_y + p["front_length"] * yaw_rate, v_x_effective) - steering
    )
    alpha_rear = np.arctan2(
        v_y - p["rear_length"] * yaw_rate, v_x_effective
    )
    force_y_front = -p["front_cornering_stiffness"] * alpha_front
    force_y_rear = -p["rear_cornering_stiffness"] * alpha_rear
    force_x = p["mass"] * acceleration
    force_x_front = p["driven_front_fraction"] * force_x
    force_x_rear = (1.0 - p["driven_front_fraction"]) * force_x
    v_x_dot = (
        force_x_front * np.cos(steering)
        - force_y_front * np.sin(steering)
        + force_x_rear
    ) / p["mass"] + yaw_rate * v_y
    v_y_dot = (
        force_x_front * np.sin(steering)
        + force_y_front * np.cos(steering)
        + force_y_rear
    ) / p["mass"] - yaw_rate * v_x
    yaw_rate_dot = (
        p["front_length"]
        * (force_x_front * np.sin(steering) + force_y_front * np.cos(steering))
        - p["rear_length"] * force_y_rear
    ) / p["yaw_inertia"]
    derivative = np.array(
        [
            v_x * np.cos(heading) - v_y * np.sin(heading),
            v_x * np.sin(heading) + v_y * np.cos(heading),
            v_x_dot,
            v_y_dot,
            yaw_rate_dot,
            yaw_rate,
        ]
    )
    next_state = state + dt * derivative

    # The linear tire equations become numerically stiff as the vehicle stops.
    # Preserve the straight-line equilibrium instead of amplifying roundoff.
    if (
        abs(v_x) <= p["minimum_forward_speed"]
        and abs(steering) <= 1e-12
        and abs(v_y) <= 1e-8
        and abs(yaw_rate) <= 1e-8
    ):
        next_state[3:5] = 0.0
    next_state[5] = wrap_angle(next_state[5])
    return next_state


def bicycle_parameters(dt):
    return {"dt": float(dt), **asdict(DEFAULT_BICYCLE_PARAMS)}
