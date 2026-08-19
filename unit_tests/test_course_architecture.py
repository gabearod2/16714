import importlib
import inspect
import unittest
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

from lecture21.value_learning import QuadraticQFunction, run_value_episode
from shared.numerical_rollout import NumericalPlant
from shared.rl import LinearGaussianActor, ScalarQuadraticCritic, collect_episode
from spark_robot import LinearDiscreteDynamicsConfig


COURSE_ROOT = Path(__file__).resolve().parents[1]


class LecturePackageTests(unittest.TestCase):
    def test_every_lecture_has_standalone_config_and_main(self):
        packages = (
            "lecture2",
            "lecture3",
            "lecture7.lqr",
            "lecture7.ilqr",
            "lecture9",
            "lecture10",
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
        for package in packages:
            with self.subTest(package=package):
                config = importlib.import_module(f"{package}.config")
                main = importlib.import_module(f"{package}.main")
                self.assertTrue(is_dataclass(config.DEFAULT_CONFIG))
                self.assertTrue(callable(main.run))
                expected_results = COURSE_ROOT.joinpath(*package.split("."), "results")
                self.assertEqual(main.DATA_OUTPUT_DIR, expected_results)

    def test_course_rl_monolith_is_removed(self):
        self.assertFalse((COURSE_ROOT / "lib" / "reinforcement_learning.py").exists())

    def test_root_compatibility_scripts_are_removed(self):
        obsolete = ["control_pipeline.py", "lecture10_mpc.py"] + [
            f"lecture{number}.py"
            for number in (2, 3, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23)
        ]
        for filename in obsolete:
            with self.subTest(filename=filename):
                self.assertFalse((COURSE_ROOT / filename).exists())


class ReinforcementLearningBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.robot_cfg = LinearDiscreteDynamicsConfig([[0.5]], [[1.0]])
        self.stage_cost = lambda state, action: float(
            (state[0] ** 2 + action[0] ** 2) / 2.0
        )

    def test_episode_collector_uses_supplied_course_plant(self):
        plant = NumericalPlant(self.robot_cfg, [1.0], dt=1.0, max_steps=3)
        episode = collect_episode(
            plant,
            lambda state: np.array([-0.25 * state[0]]),
            self.stage_cost,
            max_steps=3,
        )
        self.assertEqual(episode.states.shape, (4, 1))
        self.assertEqual(episode.actions.shape, (3, 1))

    def test_actor_critic_and_q_representations_are_independent(self):
        actor = LinearGaussianActor(-0.1, 0.1)
        critic = ScalarQuadraticCritic(3.0)
        q_function = QuadraticQFunction(np.eye(2))
        self.assertEqual(actor.parameters.shape, (2,))
        self.assertIsInstance(critic.value(np.array([1.0])), float)
        self.assertIsInstance(q_function.q_value([1.0], [0.0]), float)

    def test_value_algorithm_receives_plant_instead_of_selecting_dynamics(self):
        plant = NumericalPlant(self.robot_cfg, [1.0], dt=1.0, max_steps=2)
        q_function = QuadraticQFunction(np.array([[4.0, 1.0], [1.0, 4.0]]))
        episode = run_value_episode(
            "qlearning",
            plant,
            q_function,
            np.random.default_rng(7),
            self.stage_cost,
            learning_rate=0.1,
            epsilon=0.0,
            max_steps=2,
            stop=lambda state, step: False,
        )
        self.assertEqual(episode.length, 2)
        source = inspect.getsource(run_value_episode)
        self.assertNotIn("DynamicsConfig", source)


if __name__ == "__main__":
    unittest.main()
