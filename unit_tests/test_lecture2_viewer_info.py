import unittest

import numpy as np

from lecture2 import main as lecture2
from spark_robot import (
    AgiBotG1MobileBaseDynamic1Config,
    AgiBotG1MobileBaseUnicycleDynamic1Config,
)


class _AgentSpy:
    def __init__(self):
        self.information = []

    def set_viewer_simulation_info(self, information, *, replace=False):
        self.information.append((dict(information), replace))

    def is_running(self):
        return True

    def reset(self, _reset_info):
        pass

    def begin_render_frame(self):
        pass

    def render_line_segment(self, *_args):
        pass

    def render_sphere(self, *_args):
        pass

    def render(self):
        pass


class Lecture2ViewerInformationTests(unittest.TestCase):
    def test_replay_reports_the_trajectory_model_instead_of_viewer_model(self):
        viewer = object.__new__(lecture2.Lecture2VehicleViewer)
        viewer.config = lecture2.DEFAULT_CONFIG
        viewer.robot_cfg = AgiBotG1MobileBaseDynamic1Config()
        viewer.default_q = np.array(
            [viewer.robot_cfg.DefaultDoFVal[dof] for dof in viewer.robot_cfg.DoFs]
        )
        viewer.agent = _AgentSpy()
        model = AgiBotG1MobileBaseUnicycleDynamic1Config().create_dynamics_model()

        viewer.replay(
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05]]),
            sample_dt=0.1,
            goal=(1.0, 1.0),
            dynamics_model=model,
            integrator="Euler",
            experiment="unicycle",
        )

        initial, replace = viewer.agent.information[0]
        self.assertTrue(replace)
        self.assertEqual(
            initial["Robot config"],
            "AgiBotG1MobileBaseUnicycleDynamic1Config",
        )
        self.assertEqual(initial["Dynamics model"], "unicycle")
        self.assertEqual(initial["Dimensions"], "state=3, control=2")
        self.assertEqual(initial["Propagation"], "trajectory replay (Euler)")
        self.assertEqual(
            viewer.agent.information[-1][0]["Simulation time"],
            "0.100 s",
        )


if __name__ == "__main__":
    unittest.main()
