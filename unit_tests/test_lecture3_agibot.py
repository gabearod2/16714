import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from lecture3 import main as lecture3


class Lecture3AgiBotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.save_dir = Path(cls.temporary_directory.name)
        cls.summary = lecture3.run(
            save_dir=cls.save_dir,
            config=replace(lecture3.DEFAULT_CONFIG, steps=12),
            enable_viewer=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_uses_spark_agibot_right_arm(self):
        self.assertEqual(
            self.summary["robot_config"],
            "AgiBotG1RightArmDynamic1Config",
        )
        self.assertEqual(self.summary["end_effector_frame"], "R_ee")
        self.assertEqual(self.summary["num_joints"], 7)

    def test_generated_traces_have_agibot_joint_and_ee_coordinates(self):
        states = np.loadtxt(
            self.save_dir / "cartesian_servo_states.csv",
            delimiter=",",
            skiprows=1,
        )
        controls = np.loadtxt(
            self.save_dir / "cartesian_servo_controls.csv",
            delimiter=",",
            skiprows=1,
        )
        self.assertEqual(states.shape, (13, 11))
        self.assertEqual(controls.shape, (12, 8))
        self.assertTrue(np.isfinite(states).all())
        self.assertTrue(np.isfinite(controls).all())
        self.assertLessEqual(np.abs(controls[:, 1:]).max(), 1.5)

    def test_both_servo_laws_reduce_their_regulated_error(self):
        joint = np.loadtxt(
            self.save_dir / "joint_servo_states.csv",
            delimiter=",",
            skiprows=1,
        )
        cartesian = np.loadtxt(
            self.save_dir / "cartesian_servo_states.csv",
            delimiter=",",
            skiprows=1,
        )
        goal = np.asarray(
            self.summary["joint_vs_cartesian_servo"]["cartesian_goal"],
            dtype=float,
        )
        target_q = np.asarray(
            self.summary["joint_vs_cartesian_servo"]["joint_goal"],
            dtype=float,
        )

        self.assertLess(
            np.linalg.norm(joint[-1, 1:8] - target_q),
            np.linalg.norm(joint[0, 1:8] - target_q),
        )
        self.assertLess(
            np.linalg.norm(cartesian[-1, 8:11] - goal),
            np.linalg.norm(cartesian[0, 8:11] - goal),
        )

    def test_viewer_gripper_matches_reduced_kinematics_reference(self):
        config = lecture3.AgiBotG1RightArmDynamic1Config()
        kinematics = lecture3.AgiBotG1RightArmKinematics(config)
        agent = lecture3.AgiBotG1RightArmAgent(
            config,
            enable_viewer=False,
            enable_keyboard_control=False,
            enable_camera=False,
            use_sim_dynamics=False,
            dt=0.1,
            control_decimation=1,
            real_time=False,
            obstacle_debug={"num_obstacle": 0},
        )
        try:
            q = np.array([config.DefaultDoFVal[dof] for dof in config.DoFs])
            agent.reset({"reset_dof_pos": q})

            finger_visuals = (
                "right_narrow1_visual",
                "right_narrow3_visual",
                "right_narrow4_visual",
                "right_narrow2_visual",
                "right_wide1_visual",
                "right_wide3_visual",
                "right_wide4_visual",
                "right_wide2_visual",
            )
            gripper_center = np.mean(
                [
                    agent.data.geom_xpos[
                        mujoco.mj_name2id(
                            agent.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom_name,
                        )
                    ]
                    for geom_name in finger_visuals
                ],
                axis=0,
            )
            ee_position = kinematics.forward_kinematics(q)[
                config.Frames.R_ee, :3, 3
            ]
            # The official coupled gripper now opens asymmetrically, so the
            # mean of its visible finger meshes shifts a few millimetres from
            # the fixed 100 mm grasp-center frame.
            self.assertLess(np.linalg.norm(ee_position - gripper_center), 0.01)
        finally:
            agent.close_viewer()


if __name__ == "__main__":
    unittest.main()
