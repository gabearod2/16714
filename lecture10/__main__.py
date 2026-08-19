import argparse

from .config import DEFAULT_CONFIG
from .main import run


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="AgiBot G1 constrained MPC")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.enable_viewer,
        help="Show the SPARK MuJoCo viewer",
    )
    parser.add_argument(
        "--show-simulation-info",
        dest="viewer_show_simulation_info",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.viewer_show_simulation_info,
        help="Show the SPARK dynamics information overlay",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_cli_args(argv)
    return run(
        enable_viewer=args.viewer,
        viewer_show_simulation_info=args.viewer_show_simulation_info,
    )


if __name__ == "__main__":
    main()
