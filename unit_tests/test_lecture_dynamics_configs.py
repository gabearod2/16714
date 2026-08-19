import unittest
from types import SimpleNamespace

from spark_utils import initialize_class

from lecture7.lqr import main as lecture7_lqr
from lecture7.ilqr import main as lecture7_ilqr
from lecture9 import main as lecture9
from lecture10 import main as lecture10
from shared.control_pipeline import policy_config_components


class LectureDynamicsConfigTests(unittest.TestCase):
    def _build_policy(self, module, **kwargs):
        cfg = module.make_config(enable_viewer=False, **kwargs)
        robot_cfg = initialize_class(cfg.robot.cfg)
        _, nominal_controller, _ = policy_config_components(cfg)
        policy = initialize_class(
            nominal_controller,
            robot_cfg=robot_cfg,
            robot_kinematics=None,
        )
        return robot_cfg, policy

    def test_policy_config_components_supports_both_spark_layouts(self):
        old_nominal = object()
        old_safety = object()
        old_composition = SimpleNamespace(policy=old_nominal, safe_controller=old_safety)
        old_cfg = SimpleNamespace(algo=old_composition)

        new_nominal = object()
        new_safety = object()
        new_composition = SimpleNamespace(
            nominal_controller=new_nominal,
            safe_controller=new_safety,
        )
        new_cfg = SimpleNamespace(policy=new_composition)

        self.assertEqual(
            policy_config_components(old_cfg),
            (old_composition, old_nominal, old_safety),
        )
        self.assertEqual(
            policy_config_components(new_cfg),
            (new_composition, new_nominal, new_safety),
        )

    def test_lqr_and_mpc_examples_use_dynamic2(self):
        cases = (
            (lecture7_lqr, {}, 4, 2),
            (lecture9, {"case_name": "linear_mpc"}, 2, 1),
            (lecture10, {"case_name": "linear_mpc"}, 2, 1),
        )
        for module, kwargs, state_dim, control_dim in cases:
            with self.subTest(module=module.__name__):
                robot_cfg, policy = self._build_policy(module, **kwargs)
                self.assertEqual(type(robot_cfg).__name__, "AgiBotG1MobileBaseDynamic2Config")
                self.assertEqual(policy.dynamics_model.variant, "double_integrator")
                self.assertEqual((policy.state_dim, policy.control_dim), (state_dim, control_dim))

    def test_ilqr_example_uses_unicycle_config(self):
        robot_cfg, policy = self._build_policy(lecture7_ilqr)
        self.assertEqual(
            type(robot_cfg).__name__,
            "AgiBotG1MobileBaseUnicycleDynamic1Config",
        )
        self.assertEqual(policy.dynamics_model.variant, "unicycle")


if __name__ == "__main__":
    unittest.main()
