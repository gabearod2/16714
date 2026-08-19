#!/usr/bin/env python3
"""Validate course-repository syntax, boundaries, and release hygiene."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
OBSOLETE_ROOT_FILES = {
    "control_pipeline.py",
    "lecture10_mpc.py",
    *(f"lecture{number}.py" for number in (2, 3, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23)),
}


def tracked_files() -> tuple[Path, ...]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return tuple(ROOT / item.decode() for item in output.split(b"\0") if item)


def main() -> int:
    errors: list[str] = []

    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_dir():
            continue
        if path.name in CACHE_NAMES or path.name.endswith(".egg-info"):
            errors.append(f"local cache/build metadata remains: {path.relative_to(ROOT)}")

    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix in {".pyc", ".pyo", ".log"}:
            errors.append(f"generated file is tracked: {relative}")
        if any(part in CACHE_NAMES or part.endswith(".egg-info") for part in relative.parts):
            errors.append(f"generated directory is tracked: {relative}")

    for filename in OBSOLETE_ROOT_FILES:
        if (ROOT / filename).exists():
            errors.append(f"obsolete root compatibility script remains: {filename}")
    if (ROOT / "lib").exists():
        errors.append("obsolete local controller library remains: lib")

    homework = ROOT / "hw1"
    homework_sources = (
        "README.md",
        "problem.py",
        "solution.py",
        "vehicle_models.py",
        "unicycle_bicycle_helpers.py",
    )
    for filename in homework_sources:
        if not (homework / filename).is_file():
            errors.append(f"missing Homework 1 source: hw1/{filename}")
    for filename in ("problem.py", "solution.py"):
        path = homework / filename
        if path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) > 200:
            errors.append(f"Homework 1 entry script exceeds 200 lines: hw1/{filename}")

    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "unit_tests/README.md")
    )
    if "--branch 16714" in documentation or "aaf7b369" in documentation:
        errors.append("documentation still references the retired SPARK course branch")

    if errors:
        print("Course repository check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Course repository check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
