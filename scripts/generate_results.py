"""Regenerate lecture artifacts through canonical packaged entry points."""

from __future__ import annotations

import argparse
from importlib import import_module


NUMERICAL_LECTURES = (
    "lecture2",
    "lecture3",
    "lecture12",
    "lecture13",
    "lecture14",
    "lecture15",
    "lecture16",
    "lecture17",
    "lecture18",
    "lecture19",
    "lecture21",
    "lecture22",
    "lecture23",
)
SIMULATOR_LECTURES = (
    "lecture7.lqr",
    "lecture7.ilqr",
    "lecture9",
    "lecture10",
)
VIEWER_LECTURES = ("lecture2", "lecture3") + SIMULATOR_LECTURES


def generate(
    names,
    *,
    headless: bool = True,
    viewer_show_simulation_info: bool = True,
) -> None:
    for name in names:
        module = import_module(f"{name}.main")
        kwargs = {}
        if name in VIEWER_LECTURES:
            kwargs["enable_viewer"] = not headless
            kwargs["viewer_show_simulation_info"] = bool(
                viewer_show_simulation_info
            )
        if headless and name in VIEWER_LECTURES:
            kwargs.update(enable_viewer=False, real_time=False)
        print(f"Generating {name}")
        module.run(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*", help="Lecture package suffixes")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include simulator-backed lectures as well as numerical lectures",
    )
    parser.add_argument("--viewer", action="store_true", help="Enable simulator viewers")
    parser.add_argument(
        "--show-simulation-info",
        dest="viewer_show_simulation_info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the SPARK dynamics information overlay in enabled viewers",
    )
    args = parser.parse_args()
    names = tuple(args.lectures) or (
        NUMERICAL_LECTURES + SIMULATOR_LECTURES if args.all else NUMERICAL_LECTURES
    )
    generate(
        names,
        headless=not args.viewer,
        viewer_show_simulation_info=args.viewer_show_simulation_info,
    )


if __name__ == "__main__":
    main()
