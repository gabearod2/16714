import tempfile
import unittest
from pathlib import Path

from lecture2 import main as lecture2


class Lecture2ModelTests(unittest.TestCase):
    def test_bicycle_example_uses_dynamic2_state_and_control_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = lecture2._bicycle_example(Path(directory))
            state_header = (Path(directory) / "bicycle_dynamic_states.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            control_header = (
                Path(directory) / "bicycle_dynamic_controls.csv"
            ).read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(state_header, "t,X,Y,v_x,v_y,r,psi")
        self.assertEqual(control_header, "t,a_x,delta")
        self.assertLessEqual(summary["goal_error"], 0.1)
        self.assertLess(summary["samples"], lecture2.DEFAULT_CONFIG.bicycle_steps + 1)


if __name__ == "__main__":
    unittest.main()
