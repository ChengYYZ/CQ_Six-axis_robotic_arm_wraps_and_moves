"""Offline geometry and controller-sequence checks; never loads the robot SDK."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_c_cup_z_rotation as rotation_test
from project0714_calib.common import rpy_xyz_to_matrix


class RotationTest(unittest.TestCase):
    def test_taught_side_exact_pose_and_return(self):
        args = rotation_test.parse_args(["--taught-side"])
        targets = rotation_test.plan_targets(args)
        self.assertEqual(len(targets), 3)
        np.testing.assert_array_equal(targets[1], [627.067, 311.427, 470.629, -0.736, 3.143, 75.357])
        np.testing.assert_array_equal(targets[0], targets[2])
        self.assertNotIn("90", " ".join(rotation_test.step_labels(args)))

    def test_tilted_local_axis_and_exact_return(self):
        pose = np.array([100., 200., 300., 40., -25., 70.])
        targets = rotation_test.build_targets(pose)
        initial = rpy_xyz_to_matrix(np.radians(pose[3:]))
        for angle, target in zip(rotation_test.ANGLES_DEG, targets):
            actual = rpy_xyz_to_matrix(np.radians(target[3:]))
            np.testing.assert_allclose(target[:3], pose[:3], atol=1e-10)
            np.testing.assert_allclose(actual[:, 2], initial[:, 2], atol=1e-10)
            relative = initial.T @ actual
            np.testing.assert_allclose(relative, rpy_xyz_to_matrix(np.radians([0, 0, angle])), atol=1e-10)
        np.testing.assert_array_equal(targets[2], pose)
        np.testing.assert_array_equal(targets[4], pose)

    def test_offset_side_cup_tcp_composition(self):
        offset = np.array([102.5, -245., 113.])
        xyz, rpy = rotation_test.cup_tcp(offset, np.array([0., -90., 0.]))
        main = rpy_xyz_to_matrix(np.radians(rotation_test.TCP_RPY_DEG))
        np.testing.assert_allclose(main.T @ (xyz - rotation_test.TCP_XYZ_MM), offset, atol=1e-10)
        local = main.T @ rpy_xyz_to_matrix(np.radians(rpy))
        np.testing.assert_allclose(local[:, 2], [-1, 0, 0], atol=1e-10)

    def run_mocked(self, fail_at=None, motion="movel", conf="current"):
        args = rotation_test.parse_args(["--rotation-motion", motion, "--rotation-conf", conf])
        targets = rotation_test.build_targets(np.asarray(args.c_pose))
        robot = MagicMock()
        replies = []
        for index, target in enumerate(targets):
            pose = MagicMock()
            pose.translation_mm.return_value = target[:3]
            pose.rpy_deg_xyz.return_value = target[3:]
            replies.append(RuntimeError("unreachable") if index == fail_at else pose)
        robot.move_to_pose_mm_deg.side_effect = replies
        with patch.object(rotation_test, "XCoreRobotClient") as client, \
             patch.object(rotation_test, "confirm") as confirm, \
             patch.object(rotation_test.time, "sleep") as sleep:
            client.return_value.__enter__.return_value = robot
            if fail_at is None:
                rotation_test.execute(args, targets)
            else:
                with self.assertRaisesRegex(RuntimeError, "unreachable"):
                    rotation_test.execute(args, targets)
            if motion == "movej" and fail_at is None:
                self.assertEqual(confirm.call_count, 6)
        return robot, sleep

    def test_motion_sequence_and_arrival_holds(self):
        robot, sleep = self.run_mocked()
        calls = robot.move_to_pose_mm_deg.call_args_list
        self.assertEqual([c.kwargs["options"].motion for c in calls], ["movej"] + ["movel"] * 4)
        self.assertEqual([c.kwargs["options"].use_current_conf_data for c in calls], [False] + [True] * 4)
        self.assertEqual([c.args[0] for c in robot.set_conf_data_forced.call_args_list],
                         [False] + [True] * 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertTrue(all(c.args == (1.0,) for c in sleep.call_args_list))
        robot.stop_motion.assert_not_called()

    def test_failed_rotation_stops_without_return_move(self):
        robot, sleep = self.run_mocked(fail_at=1)
        self.assertEqual(robot.move_to_pose_mm_deg.call_count, 2)
        robot.stop_motion.assert_called_once()
        self.assertEqual(sleep.call_count, 1)

    def test_explicit_movej_preserves_rotation_configuration(self):
        robot, _ = self.run_mocked(motion="movej")
        calls = robot.move_to_pose_mm_deg.call_args_list
        self.assertTrue(all(c.kwargs["options"].motion == "movej" for c in calls))
        self.assertTrue(all(c.kwargs["options"].use_current_conf_data for c in calls[1:]))

    def test_dry_run_never_connects(self):
        with patch.object(rotation_test, "XCoreRobotClient") as client:
            self.assertEqual(rotation_test.main([]), 0)
        client.assert_not_called()

    def test_auto_conf_is_explicit_and_only_for_movej(self):
        with self.assertRaises(SystemExit):
            rotation_test.parse_args(["--rotation-conf", "auto"])
        robot, _ = self.run_mocked(motion="movej", conf="auto")
        self.assertTrue(all(not c.kwargs["options"].use_current_conf_data
                            for c in robot.move_to_pose_mm_deg.call_args_list))
        self.assertEqual([c.args[0] for c in robot.set_conf_data_forced.call_args_list], [False] * 5)

    def test_controller_conf_switch_checks_error(self):
        from project0714_calib.xcore_robot import XCoreRobotClient
        client = XCoreRobotClient.__new__(XCoreRobotClient)
        client.connected = True
        client.ec = {}
        client.robot = MagicMock()
        client.xcore = MagicMock()
        client.set_conf_data_forced(False)
        client.robot.setDefaultConfOpt.assert_called_once_with(False, client.ec)
        def reject(forced, ec):
            ec["code"] = -1
        client.robot.setDefaultConfOpt.side_effect = reject
        client.xcore.message.return_value = "rejected"
        with self.assertRaisesRegex(RuntimeError, "setDefaultConfOpt failed"):
            client.set_conf_data_forced(True)

    def test_auto_conf_failure_does_not_retry(self):
        robot, _ = self.run_mocked(fail_at=1, motion="movej", conf="auto")
        self.assertEqual(robot.move_to_pose_mm_deg.call_count, 2)
        robot.stop_motion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
