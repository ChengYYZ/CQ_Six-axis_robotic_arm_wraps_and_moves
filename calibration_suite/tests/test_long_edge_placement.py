from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

import surface_cluster_grasp as grasp
from project0714_calib.common import rpy_xyz_to_matrix
from surface_cluster_grasp import (
    FIXED_GRASP_X_YAW_DEG,
    WAYPOINT_D,
    ClusterCandidate,
    SuctionCupSpec,
    build_parser,
    compute_grasp_rpy_from_normal_and_fixed_x,
    placement_local_z_delta_deg,
    placement_aligned_pickup_rpy_candidates,
    placement_tcp_waypoint_for_selected_cup,
    require_selected_cup_at_functional_waypoint,
    rotate_selected_cup_at_functional_waypoint,
    rotate_waypoint_about_local_z,
    target_rpy_for_candidate,
    unwrap_rpy_deg,
    waypoint_for_selected_cup,
    wrap_undirected_angle_deg,
)


class LongEdgePlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.d_rotation = rpy_xyz_to_matrix(
            np.radians([WAYPOINT_D.rx_deg, WAYPOINT_D.ry_deg, WAYPOINT_D.rz_deg])
        )

    def test_angle_is_undirected_and_uses_shortest_turn(self) -> None:
        self.assertAlmostEqual(wrap_undirected_angle_deg(100.0), -80.0)
        self.assertAlmostEqual(wrap_undirected_angle_deg(-100.0), 80.0)
        self.assertAlmostEqual(wrap_undirected_angle_deg(180.0), 0.0)

    def test_rpy_unwrap_prevents_270_degree_wrist_command(self) -> None:
        continuous = unwrap_rpy_deg(
            np.asarray([0.0, 0.0, 175.38]),
            np.asarray([-1.917, 2.748, -94.661]),
        )
        self.assertAlmostEqual(continuous[2], -184.62, places=6)
        self.assertLess(abs(continuous[2] - (-94.661)), 91.0)

    def test_return_to_a_star_uses_nearest_equivalent_angle(self) -> None:
        current_pose = SimpleNamespace(
            rpy_deg_xyz=lambda: np.asarray([0.0, 0.0, 173.91], dtype=np.float64)
        )
        robot = SimpleNamespace(read_current_pose=lambda: current_pose)
        motion_options = grasp.MotionOptions(motion="movej", use_current_conf_data=True)

        with patch.object(grasp, "move_pose_with_singularity_fallback") as move:
            grasp.move_waypoint(grasp.WAYPOINT_A_STAR, robot, motion_options)

        commanded_rz_deg = move.call_args.args[6]
        self.assertAlmostEqual(commanded_rz_deg, 265.339, places=6)
        self.assertLess(abs(commanded_rz_deg - 173.91), 92.0)

    def test_operator_stop_is_not_a_retryable_plan_failure(self) -> None:
        self.assertTrue(
            grasp.is_operator_software_stop_error(
                RuntimeError("Robot motion interrupted by operator software stop request.")
            )
        )

    def test_loaded_transfer_also_uses_nearest_equivalent_angle(self) -> None:
        current_pose = SimpleNamespace(
            rpy_deg_xyz=lambda: np.asarray([0.0, 0.0, 173.91], dtype=np.float64)
        )
        robot = SimpleNamespace(read_current_pose=lambda: current_pose)
        motion_options = grasp.MotionOptions(motion="movel", use_current_conf_data=True)

        with patch.object(grasp, "move_pose_with_singularity_fallback") as move:
            grasp.move_loaded_transfer_with_singularity_fallback(
                grasp.WAYPOINT_A_STAR,
                robot,
                motion_options,
                allow_current_conf_movej_fallback=False,
            )

        commanded_rz_deg = move.call_args.args[6]
        self.assertAlmostEqual(commanded_rz_deg, 265.339, places=6)
        self.assertLess(abs(commanded_rz_deg - 173.91), 92.0)

    def test_functional_c_prefers_short_turn_over_secondary_reach_score(self) -> None:
        secondary = SuctionCupSpec(
            name="secondary",
            do_port=5,
            offset_tool_mm=np.asarray([0.0, -245.0, 0.0], dtype=np.float64),
        )
        candidates = grasp.functional_waypoint_candidates_for_cup(
            grasp.WAYPOINT_C,
            secondary,
            np.asarray([779.155, -218.594, 487.419], dtype=np.float64),
            np.asarray([-0.296, 1.428, -6.094], dtype=np.float64),
            prefer_original_orientation=False,
        )

        self.assertEqual(candidates[0].name, "C[secondary]")
        self.assertNotIn("yaw+120", candidates[0].name)
        first_rotation = rpy_xyz_to_matrix(
            np.radians([candidates[0].rx_deg, candidates[0].ry_deg, candidates[0].rz_deg])
        )
        current_rotation = rpy_xyz_to_matrix(np.radians([-0.296, 1.428, -6.094]))
        self.assertLess(grasp.rotation_distance_deg(current_rotation, first_rotation), 10.0)

    def test_loaded_movel_singularity_retries_movej_with_current_confdata(self) -> None:
        fallback_calls = []
        current_pose = SimpleNamespace(
            rpy_deg_xyz=lambda: np.asarray([-0.296, 1.428, -6.094], dtype=np.float64)
        )
        robot = SimpleNamespace(
            read_current_pose=lambda: current_pose,
            stop_motion=lambda: None,
            move_to_pose_mm_deg=lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
        )
        motion_options = grasp.MotionOptions(motion="movel", use_current_conf_data=True)

        with patch.object(
            grasp,
            "move_pose_with_singularity_fallback",
            side_effect=RuntimeError(
                "Robot move rejected before start: {'ec': -50102, 'message': '存在穿越奇异点的轨迹'}"
            ),
        ):
            grasp.move_loaded_transfer_with_singularity_fallback(
                grasp.WAYPOINT_C,
                robot,
                motion_options,
            )

        self.assertEqual(len(fallback_calls), 1)
        fallback_options = fallback_calls[0][1]["options"]
        self.assertEqual(fallback_options.motion, "movej")
        self.assertTrue(fallback_options.use_current_conf_data)

    def test_default_pickup_keeps_suction_face_horizontal(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.normal_mode, "base-z")
        pickup_rpy = compute_grasp_rpy_from_normal_and_fixed_x(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            FIXED_GRASP_X_YAW_DEG,
            "minus-z",
        )
        np.testing.assert_allclose(pickup_rpy[:2], [0.0, 0.0], atol=1e-9)

    def test_pickup_preorientation_eliminates_loaded_d_rotation(self) -> None:
        package_long = np.asarray([-0.4049, -0.9144, 0.0], dtype=np.float64)
        package_long /= np.linalg.norm(package_long)
        package_short = np.asarray(
            [package_long[1], -package_long[0], 0.0], dtype=np.float64
        )
        candidate = ClusterCandidate(
            index=1,
            center_pixel=(0, 0),
            hull_pixels=np.zeros((4, 2), dtype=np.int32),
            support_region_id="test",
            support_region_name="test",
            point_camera_mm=np.zeros(3),
            point_base_mm=np.zeros(3),
            normal_camera=np.asarray([0.0, 0.0, 1.0]),
            normal_base=np.asarray([0.0, 0.0, 1.0]),
            height_mm=0.0,
            flatness_mm=0.0,
            point_count=100,
            short_axis_base=package_short,
        )
        current_pose = SimpleNamespace(
            rpy_rad_xyz=np.radians([0.0, 0.0, -6.094])
        )
        args = build_parser().parse_args([])
        candidates = placement_aligned_pickup_rpy_candidates(candidate, current_pose, args)
        self.assertEqual(len(candidates), 2)

        pickup_rotation = rpy_xyz_to_matrix(np.radians(candidates[0][1]))
        delta_deg = placement_local_z_delta_deg(
            package_long,
            pickup_rotation,
            self.d_rotation,
        )
        self.assertAlmostEqual(delta_deg, 0.0, places=7)

        selected_rpy, selected_mode = target_rpy_for_candidate(
            candidate,
            current_pose,
            args,
        )
        self.assertIsInstance(selected_rpy, np.ndarray)
        self.assertEqual(selected_rpy.shape, (3,))
        self.assertIsInstance(selected_mode, str)
        self.assertTrue(selected_mode.startswith("align_normal_package_long_for_d"))

    def test_package_long_edge_reaches_platform_long_direction(self) -> None:
        pickup_rotation = rpy_xyz_to_matrix(np.radians([0.0, 0.0, -96.09]))
        platform_long_xy = self.d_rotation[:2, 1]
        platform_long_xy /= np.linalg.norm(platform_long_xy)

        for package_angle_deg in (-80.0, -45.0, 0.0, 35.0, 80.0):
            angle_rad = np.radians(package_angle_deg)
            package_long_base = np.asarray(
                [np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=np.float64
            )
            delta_deg = placement_local_z_delta_deg(
                package_long_base,
                pickup_rotation,
                self.d_rotation,
            )
            target_rotation = self.d_rotation @ rpy_xyz_to_matrix(
                np.radians([0.0, 0.0, delta_deg])
            )
            predicted_long_base = target_rotation @ (
                pickup_rotation.T @ package_long_base
            )
            predicted_long_xy = predicted_long_base[:2]
            predicted_long_xy /= np.linalg.norm(predicted_long_xy)

            self.assertLessEqual(abs(delta_deg), 90.0)
            self.assertAlmostEqual(
                abs(float(np.dot(predicted_long_xy, platform_long_xy))),
                1.0,
                places=8,
            )

    def test_local_z_rotation_preserves_package_bottom_reference_normal(self) -> None:
        reference_z = self.d_rotation[:, 2]
        for delta_deg in (-90.0, -30.0, 0.0, 45.0, 89.0):
            rotated = rotate_waypoint_about_local_z(WAYPOINT_D, delta_deg)
            rotated_matrix = rpy_xyz_to_matrix(
                np.radians([rotated.rx_deg, rotated.ry_deg, rotated.rz_deg])
            )
            np.testing.assert_allclose(rotated_matrix[:, 2], reference_z, atol=1e-9)

    def test_secondary_cup_center_remains_at_aligned_d(self) -> None:
        secondary = SuctionCupSpec(
            name="secondary",
            do_port=5,
            offset_tool_mm=np.asarray([0.0, -245.0, 0.0], dtype=np.float64),
        )
        for delta_deg in (-90.0, -35.0, 0.0, 62.0, 89.0):
            aligned_d = rotate_waypoint_about_local_z(WAYPOINT_D, delta_deg)
            compensated_tcp = waypoint_for_selected_cup(aligned_d, secondary)
            actual_center = require_selected_cup_at_functional_waypoint(
                compensated_tcp,
                aligned_d,
                secondary,
            )
            np.testing.assert_allclose(
                actual_center,
                [aligned_d.x_mm, aligned_d.y_mm, aligned_d.z_mm],
                atol=1e-9,
            )

    def test_secondary_rotation_is_split_into_compensated_steps(self) -> None:
        secondary = SuctionCupSpec(
            name="secondary",
            do_port=5,
            offset_tool_mm=np.asarray([0.0, -245.0, 0.0], dtype=np.float64),
        )
        commanded_physical_waypoints = []

        def fake_move(physical_waypoint, cup, _robot, _motion_options, **_kwargs):
            commanded_physical_waypoints.append(physical_waypoint)
            tcp_waypoint = waypoint_for_selected_cup(physical_waypoint, cup)
            center = require_selected_cup_at_functional_waypoint(
                tcp_waypoint,
                physical_waypoint,
                cup,
            )
            return tcp_waypoint, center

        with patch.object(
            grasp,
            "move_selected_cup_to_functional_waypoint",
            side_effect=fake_move,
        ):
            rotate_selected_cup_at_functional_waypoint(
                WAYPOINT_D,
                90.0,
                secondary,
                object(),
                object(),
                max_step_deg=10.0,
            )

        self.assertEqual(len(commanded_physical_waypoints), 9)
        final_rotation = rpy_xyz_to_matrix(
            np.radians(
                [
                    commanded_physical_waypoints[-1].rx_deg,
                    commanded_physical_waypoints[-1].ry_deg,
                    commanded_physical_waypoints[-1].rz_deg,
                ]
            )
        )
        expected_final = self.d_rotation @ rpy_xyz_to_matrix(
            np.radians([0.0, 0.0, 90.0])
        )
        np.testing.assert_allclose(final_rotation, expected_final, atol=1e-9)

    def test_secondary_placement_keeps_taught_tcp_xy_and_matches_d_height(self) -> None:
        secondary = SuctionCupSpec(
            name="secondary",
            do_port=5,
            offset_tool_mm=np.asarray([0.0, -245.0, 0.0], dtype=np.float64),
        )
        aligned_d = rotate_waypoint_about_local_z(WAYPOINT_D, -50.49)
        tcp_waypoint, cup_center = placement_tcp_waypoint_for_selected_cup(
            aligned_d,
            secondary,
        )

        self.assertAlmostEqual(tcp_waypoint.x_mm, WAYPOINT_D.x_mm)
        self.assertAlmostEqual(tcp_waypoint.y_mm, WAYPOINT_D.y_mm)
        self.assertAlmostEqual(cup_center[2], WAYPOINT_D.z_mm, places=8)
        self.assertLess(
            np.linalg.norm([tcp_waypoint.x_mm, tcp_waypoint.y_mm, tcp_waypoint.z_mm]),
            1000.0,
        )


if __name__ == "__main__":
    unittest.main()
