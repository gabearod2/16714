"""Shared artifact writers for course lecture experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


COURSE_ROOT = Path(__file__).resolve().parents[1]


def data_dir(lecture_name: str) -> Path:
    """Return the package-local, git-ignored results directory for a lecture."""

    return COURSE_ROOT / Path(str(lecture_name)) / "results"

Array = np.ndarray

def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_csv(path: Path | str, header: tuple[str, ...] | list[str], data: Array) -> None:
    np.savetxt(path, np.asarray(data, dtype=float), delimiter=",", header=",".join(header), comments="")


def write_json(path: Path | str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
        handle.write("\n")


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value
