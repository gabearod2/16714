import tempfile
import unittest

import numpy as np
from spark_robot import (
    AgiBotG1MobileBaseBicycleDynamic2Config,
    AgiBotG1MobileBaseDynamic1Config,
)

from hw1.solution import (
    ComparisonPipeline,
    Config,
    run_gain_study,
)
from hw1 import problem, solution
from hw1.unicycle_bicycle_helpers import ModelViewer


class _ViewerAgentSpy:
    def __init__(self):
        self.information = []
        self.reset_count = 0

    def set_viewer_simulation_info(self, information, *, replace=False):
        self.information.append((dict(information), replace))

    def is_running(self):
        return True

    def reset(self, _reset_info):
        self.reset_count += 1

    def begin_render_frame(self):
        pass

    def render_line_segment(self, *_args):
        pass

    def render_sphere(self, *_args):
        pass

    def render(self):
        pass


class Homework1BicycleTests(unittest.TestCase):
    def test_released_scaffold_maps_to_solution_api(self):
        self.assertEqual(
            tuple(problem.Config.__dataclass_fields__),
            tuple(solution.Config.__dataclass_fields__),
        )
        for name in ("unicycle_control", "bicycle_initial_state", "bicycle_control",
                     "run_gain_study"):
            self.assertTrue(callable(getattr(problem, name)))
            self.assertTrue(callable(getattr(solution, name)))
        with self.assertRaisesRegex(NotImplementedError, "Problem 2.1"):
            problem.unicycle_control(np.zeros(4), problem.Config())

    def test_bicycle_uses_course_state_and_control_order(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ComparisonPipeline(Config(duration=0.1), directory)

        self.assertEqual(
            pipeline.bicycle_model.state_names,
            ("X", "Y", "v_x", "v_y", "r", "psi"),
        )
        self.assertEqual(pipeline.bicycle_model.control_names, ("a_x", "delta"))

    def test_local_euler_transition_matches_spark_euler_step(self):
        config = Config(duration=0.1)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ComparisonPipeline(config, directory)
        state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.2])
        spark_model = AgiBotG1MobileBaseBicycleDynamic2Config().create_dynamics_model()

        np.testing.assert_allclose(
            pipeline.bicycle_model.step(state, control, config.dt, "native"),
            spark_model.step(state, control, config.dt, "Euler"),
            atol=1e-12,
        )

    def test_solution_uses_canonical_unicycle_to_bicycle_control(self):
        config = Config()
        state = np.array([0.0, 0.0, 0.5, 0.1, 0.2, 0.0])
        v_dot, theta_dot = solution.unicycle_control(
            [state[0], state[1], state[2], state[5]], config
        )
        wheelbase = (
            solution.DEFAULT_BICYCLE_PARAMS.front_length
            + solution.DEFAULT_BICYCLE_PARAMS.rear_length
        )

        a_x, delta = solution.bicycle_control(state, config)

        self.assertEqual(a_x, v_dot)
        self.assertEqual(
            delta,
            np.clip(
                np.arctan2(wheelbase * theta_dot, state[2]),
                -solution.DEFAULT_BICYCLE_PARAMS.steering_limit,
                solution.DEFAULT_BICYCLE_PARAMS.steering_limit,
            ),
        )

    def test_stopped_bicycle_does_not_amplify_lateral_roundoff(self):
        config = Config(duration=0.1)
        with tempfile.TemporaryDirectory() as directory:
            model = ComparisonPipeline(config, directory).bicycle_model
        state = np.array([0.0, 0.0, 0.09, 1e-10, 1e-10, 0.0])

        next_state = model.step(state, np.zeros(2), config.dt, "native")

        np.testing.assert_array_equal(next_state[3:5], np.zeros(2))

    def test_default_bicycle_lateral_velocity_decays(self):
        config = Config()
        with tempfile.TemporaryDirectory() as directory:
            _, times, states, _ = ComparisonPipeline(config, directory).simulate(
                "bicycle"
            )

        self.assertGreater(np.max(states[:, 3]), 0.1)
        self.assertLess(np.max(np.abs(states[times >= 5.0, 3])), 1e-6)

    def test_viewer_decimates_playback_and_updates_simulation_time(self):
        viewer = object.__new__(ModelViewer)
        viewer.config = Config(duration=0.04)
        viewer.robot_config = AgiBotG1MobileBaseDynamic1Config()
        viewer.default_q = np.array(
            [viewer.robot_config.DefaultDoFVal[dof] for dof in viewer.robot_config.DoFs]
        )
        viewer.agent = _ViewerAgentSpy()
        with tempfile.TemporaryDirectory() as directory:
            model = ComparisonPipeline(viewer.config, directory).bicycle_model
        states = np.zeros((5, 6))

        viewer.replay(states, model)

        self.assertEqual(viewer.agent.reset_count, 3)
        initial_information, replace = viewer.agent.information[0]
        self.assertTrue(replace)
        self.assertEqual(initial_information["Dynamics model"], "bicycle_dynamic_6_state")
        self.assertEqual(initial_information["Robot config"], "DiscreteTimeDynamicsConfig")
        self.assertEqual(initial_information["Propagation"], "external forward Euler replay")
        self.assertEqual(viewer.agent.information[-1][0]["Simulation time"], "0.040 s")

    def test_models_run_separately_and_save_independent_results(self):
        config = Config(duration=0.02)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ComparisonPipeline(config, directory)
            unicycle = pipeline.run("unicycle")
            bicycle = pipeline.run("bicycle")

            self.assertEqual(unicycle["states"].shape, (3, 4))
            self.assertEqual(bicycle["states"].shape, (3, 6))
            self.assertEqual(
                (unicycle["output_dir"] / "states.csv").read_text().splitlines()[0],
                "time,p_x,p_y,v,theta",
            )
            self.assertEqual(
                (bicycle["output_dir"] / "states.csv").read_text().splitlines()[0],
                "time,X,Y,v_x,v_y,r,psi",
            )
            self.assertEqual(len(list(unicycle["output_dir"].glob("*.png"))), 5)
            self.assertEqual(len(list(bicycle["output_dir"].glob("*.png"))), 7)

    def test_gain_study_saves_all_controller_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_gain_study(Config(duration=0.02), directory)
            self.assertEqual(len(result["traces"]), 9)
            self.assertTrue((result["output_dir"] / "errors.csv").is_file())
            self.assertTrue((result["output_dir"] / "position_error.png").is_file())
            self.assertTrue((result["output_dir"] / "orientation_error.png").is_file())


if __name__ == "__main__":
    unittest.main()
