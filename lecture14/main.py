from pathlib import Path
from dataclasses import asdict, replace

import numpy as np

from shared.artifacts import data_dir, ensure_dir, write_csv, write_json
from .config import DEFAULT_CONFIG, Lecture14Config


DATA_OUTPUT_DIR = data_dir("lecture14")


def run(save_dir=DATA_OUTPUT_DIR, seed=None, config: Lecture14Config = DEFAULT_CONFIG):
    if seed is not None:
        config = replace(config, seed=int(seed))
    save_dir = ensure_dir(save_dir)
    rng = np.random.default_rng(config.seed)

    n = config.samples
    x = rng.standard_normal((2, n))
    AY = np.array([[1.0, 0.0]])
    AZ = np.array([[1.0, 1.0]])
    y = AY @ x
    z = AZ @ x

    x_hat_y = AY.T @ y / float(AY @ AY.T)
    z_hat_y = AZ @ AY.T @ y / float(AY @ AY.T)
    z_var_y = float(AZ @ AZ.T - AZ @ AY.T @ AY @ AZ.T / float(AY @ AY.T))
    z_tilde_y = z - z_hat_y
    x_hat_yz = x_hat_y + (AZ.T - AY.T @ AY @ AZ.T / float(AY @ AY.T)) @ z_tilde_y / z_var_y

    err_y = np.max(np.abs(x - x_hat_y), axis=1)
    err_yz = np.max(np.abs(x - x_hat_yz), axis=1)
    samples = np.column_stack(
        [
            np.arange(n),
            x.T,
            y.reshape(-1),
            z.reshape(-1),
            x_hat_y.T,
            x_hat_yz.T,
        ]
    )
    write_csv(
        save_dir / "least_squares_samples.csv",
        ("sample", "x1", "x2", "y", "z", "xhat_y1", "xhat_y2", "xhat_yz1", "xhat_yz2"),
        samples,
    )
    write_json(save_dir / "config.json", asdict(config))
    write_json(save_dir / "summary.json", {"max_error_y": err_y, "max_error_yz": err_yz, "seed": config.seed})
    return {"max_error_y": err_y, "max_error_yz": err_yz}


if __name__ == "__main__":
    run()
