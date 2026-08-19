"""Compare generated lecture summaries with compact committed baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.artifacts import COURSE_ROOT


BASELINE_PATH = COURSE_ROOT / "unit_tests" / "baselines" / "summaries.json"


def _summary_path(key: str) -> Path:
    """Resolve a baseline key to a lecture package's local results folder."""

    parts = Path(key).parts
    if len(parts) >= 2 and parts[0] == "lecture7" and parts[1] in ("lqr", "ilqr"):
        package_parts = parts[:2]
        result_parts = parts[2:]
    else:
        package_parts = parts[:1]
        result_parts = parts[1:]
    return COURSE_ROOT.joinpath(*package_parts, "results", *result_parts, "summary.json")


def _compare(expected, actual, path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors.extend(_compare(value, actual[key], f"{path}.{key}"))
    elif isinstance(expected, list):
        if not np.allclose(np.asarray(expected), np.asarray(actual), rtol=1e-7, atol=1e-9):
            errors.append(f"{path}: expected {expected}, got {actual}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, rtol=1e-7, atol=1e-9):
            errors.append(f"{path}: expected {expected}, got {actual}")
    elif expected != actual:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    return errors


def main() -> None:
    with BASELINE_PATH.open() as handle:
        baselines = json.load(handle)
    errors: list[str] = []
    for lecture, expected in baselines.items():
        summary_path = _summary_path(lecture)
        if not summary_path.exists():
            errors.append(f"{lecture}: missing {summary_path}")
            continue
        with summary_path.open() as handle:
            actual = json.load(handle)
        errors.extend(_compare(expected, actual, lecture))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(baselines)} lecture summaries")


if __name__ == "__main__":
    main()
