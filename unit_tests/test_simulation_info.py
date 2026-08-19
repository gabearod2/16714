import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lecture2.config import DEFAULT_CONFIG as LECTURE2_CONFIG
from lecture3.config import DEFAULT_CONFIG as LECTURE3_CONFIG
from lecture7.lqr import main as lecture7_lqr
from lecture7.ilqr import main as lecture7_ilqr
from lecture9 import main as lecture9
from lecture10 import main as lecture10
from scripts import generate_results


CLI_MODULES = (
    "lecture2.__main__",
    "lecture3.__main__",
    "lecture7.lqr.__main__",
    "lecture7.ilqr.__main__",
    "lecture9.__main__",
    "lecture10.__main__",
)

PIPELINE_MODULES = (
    lecture7_lqr,
    lecture7_ilqr,
    lecture9,
    lecture10,
)


class SimulationInformationTests(unittest.TestCase):
    def test_course_viewer_configs_opt_in_by_default(self):
        self.assertTrue(LECTURE2_CONFIG.viewer_show_simulation_info)
        self.assertTrue(LECTURE3_CONFIG.viewer_show_simulation_info)

        for module in PIPELINE_MODULES:
            with self.subTest(module=module.__name__):
                config = module.make_config(enable_viewer=False)
                self.assertTrue(config.env.agent.viewer_show_simulation_info)

                hidden = module.make_config(
                    enable_viewer=False,
                    viewer_show_simulation_info=False,
                )
                self.assertFalse(hidden.env.agent.viewer_show_simulation_info)

    def test_every_viewer_cli_exposes_show_simulation_info(self):
        for module_name in CLI_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(
                    module._parse_cli_args([]).viewer_show_simulation_info
                )
                self.assertTrue(
                    module._parse_cli_args(
                        ["--show-simulation-info"]
                    ).viewer_show_simulation_info
                )
                self.assertFalse(
                    module._parse_cli_args(
                        ["--no-show-simulation-info"]
                    ).viewer_show_simulation_info
                )

    def test_aggregate_generator_forwards_viewer_information_option(self):
        lecture2_run = Mock()
        lecture12_run = Mock()
        modules = {
            "lecture2.main": SimpleNamespace(run=lecture2_run),
            "lecture12.main": SimpleNamespace(run=lecture12_run),
        }
        with patch.object(
            generate_results,
            "import_module",
            side_effect=modules.__getitem__,
        ):
            generate_results.generate(
                ("lecture2", "lecture12"),
                headless=False,
                viewer_show_simulation_info=False,
            )

        lecture2_run.assert_called_once_with(
            enable_viewer=True,
            viewer_show_simulation_info=False,
        )
        lecture12_run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
