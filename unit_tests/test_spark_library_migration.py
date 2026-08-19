import unittest
from pathlib import Path

from spark_policy.control.adaptive import ModelReferenceAdaptiveController
from spark_policy.control.iterative_learning import FrequencyDomainILC, TimeDomainILC
from spark_policy.estimation import KalmanFilterEstimator
from spark_agent import DynamicsExecutor, DynamicsModelAgent
from shared.numerical_rollout import NumericalPlant, rollout_model

from lecture12 import main as lecture12
from lecture13 import main as lecture13


class SparkLibraryMigrationTests(unittest.TestCase):
    def test_reusable_implementations_resolve_from_spark_policy(self):
        for implementation in (
            KalmanFilterEstimator,
            ModelReferenceAdaptiveController,
            FrequencyDomainILC,
            TimeDomainILC,
            DynamicsExecutor,
            DynamicsModelAgent,
        ):
            with self.subTest(implementation=implementation.__name__):
                self.assertTrue(implementation.__module__.startswith("spark_"))

        for implementation in (NumericalPlant, rollout_model):
            with self.subTest(implementation=implementation.__name__):
                self.assertTrue(implementation.__module__.startswith("shared."))

    def test_obsolete_course_modules_are_removed(self):
        lib_dir = Path(__file__).resolve().parents[1] / "lib"
        for filename in ("estimators.py", "adaptive.py", "ilc.py", "rollout.py"):
            with self.subTest(filename=filename):
                self.assertFalse((lib_dir / filename).exists())

    def test_ilc_lecture_scenarios_still_run(self):
        frequency = lecture12._run_frequency_ilc(horizon=5, n_iter=2)
        time_domain = lecture13._run_time_domain_ilc(horizon=5, n_iter=2)
        self.assertEqual(frequency["states"].shape, (3, 6, 2))
        self.assertEqual(time_domain["states"].shape, (3, 6, 2))


if __name__ == "__main__":
    unittest.main()
