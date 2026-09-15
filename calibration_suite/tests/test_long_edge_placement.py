from __future__ import annotations

import sys
import socket
import threading
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
    angles_toward_fallback_deg,
    build_parser,
    barcode_side_view_waypoint,
    compute_grasp_rpy_from_normal_and_fixed_x,
    interpolated_orientation_steps_deg,
    placement_local_z_delta_deg,
    placement_alignment_error_deg,
    placement_aligned_pickup_rpy_candidates,
    placement_tcp_waypoint_for_selected_cup,
    receive_barcode_from_tcp_client,
    require_selected_cup_at_functional_waypoint,
    rotate_selected_cup_at_functional_waypoint,
    rotate_waypoint_about_local_z,
    return_empty_from_d_via_rotation_safe,
    should_inspect_c_side_view,
    target_rpy_for_candidate,
    unwrap_rpy_deg,
    verified_placement_angles_for_cup,
    verified_placement_fallback_angles_deg,
    waypoint_for_selected_cup,
    wrap_undirected_angle_deg,
)


class LongEdgePlacementTests(unittest.TestCase):
    def test_verified_fallbacks_are_ranked_by_alignment_not_interpolated(self) -> None:
        angles = verified_placement_fallback_angles_deg(
            requested_deg=74.3,
            current_deg=51.66,
            verified_angles_deg=[0.0, 90.0, 45.0, 45.0],
        )

        self.assertEqual(angles, [90.0, 45.0, 0.0])
        self.assertNotIn(49.66, angles)

    def test_placement_alignment_error_is_undirected(self) -> None:
        self.assertAlmostEqual(placement_alignment_error_deg(74.3, 0.0), 74.3)
        self.assertAlmostEqual(placement_alignment_error_deg(179.0, 1.0), 2.0)

    def test_placement_safety_defaults_require_explicit_degraded_opt_in(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(list(args.verified_placement_angles_deg), [0.0])
        self.assertEqual(args.max_placement_alignment_error_deg, 5.0)
        self.assertFalse(args.allow_degraded_placement)

    def test_primary_and_secondary_use_independent_verified_angles(self) -> None:
        args = build_parser().parse_args([
            "--primary-verified-placement-angles-deg", "0", "45",
            "--secondary-verified-placement-angles-deg", "0", "20",
        ])

        self.assertEqual(verified_placement_angles_for_cup(args, "primary"), [0.0, 45.0])
        self.assertEqual(verified_placement_angles_for_cup(args, "secondary"), [0.0, 20.0])

    def test_d_reachability_search_tries_closest_smaller_angles_first(self) -> None:
        angles = angles_toward_fallback_deg(42.44, 0.0, step_deg=2.0)

        self.assertAlmostEqual(angles[0], 40.44)
        self.assertAlmostEqual(angles[1], 38.44)
        self.assertAlmostEqual(angles[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(angles, angles[1:])))

    def test_restart_orientation_bridge_splits_large_d_to_b_rotation(self) -> None:
        start = np.asarray([-1.21, 2.90, 131.56])
        target = np.asarray([-0.30, 1.43, -6.09])

        steps = interpolated_orientation_steps_deg(start, target, max_step_deg=30.0)

        self.assertGreater(len(steps), 1)
        previous = start
        for step in steps:
            self.assertLessEqual(
                grasp.rotation_distance_deg(
                    rpy_xyz_to_matrix(np.radians(previous)),
                    rpy_xyz_to_matrix(np.radians(step)),
                ),
                30.000001,
            )
            previous = step
        np.testing.assert_allclose(
            rpy_xyz_to_matrix(np.radians(steps[-1])),
            rpy_xyz_to_matrix(np.radians(target)),
            atol=1e-9,
        )

    def test_empty_d_return_lifts_then_unwinds_at_rotation_safe_point(self) -> None:
        cup = SuctionCupSpec("primary", 6, np.zeros(3))
        calls: list[tuple[str, float]] = []

        with patch.object(
            grasp,
            "move_selected_cup_to_functional_waypoint",
            side_effect=lambda waypoint, *_args, **_kwargs: calls.append(
                ("lift", waypoint.rz_deg)
            ),
        ), patch.object(
            grasp,
            "rotate_selected_cup_at_functional_waypoint",
            side_effect=lambda _waypoint, target, *_args, **kwargs: calls.append(
                ("unwind", kwargs["start_local_z_deg"] - target)
            ),
        ):
            return_empty_from_d_via_rotation_safe(
                cup,
                48.0,
                object(),
                object(),
                on_safe_arrival=lambda: calls.append(("listen", 0.0)),
            )

        self.assertEqual(
            [name for name, _value in calls],
            ["lift", "listen", "unwind"],
        )
        self.assertAlmostEqual(calls[2][1], 48.0)

    def test_c_side_view_is_only_needed_when_bottom_and_back_have_no_waybill(self) -> None:
        self.assertTrue(should_inspect_c_side_view(None))
        self.assertTrue(
            should_inspect_c_side_view(SimpleNamespace(has_waybill=False, barcode=None))
        )
        self.assertFalse(
            should_inspect_c_side_view(SimpleNamespace(has_waybill=True, barcode=None))
        )
        self.assertFalse(
            should_inspect_c_side_view(
                SimpleNamespace(has_waybill=True, barcode="PKG-123456")
            )
        )

    def test_barcode_reader_network_defaults_match_camera_configuration(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.barcode_reader_bind_ip, "192.168.2.100")
        self.assertEqual(args.barcode_reader_port, 3001)
        self.assertEqual(args.place_dwell_s, 2.0)
        self.assertEqual(args.barcode_reader_timeout_s, 3.0)

    def test_barcode_reader_accepts_tcp_client_payload(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        result: list[str | None] = []
        receiver = threading.Thread(
            target=lambda: result.append(
                receive_barcode_from_tcp_client("127.0.0.1", port, 1.0)
            )
        )
        receiver.start()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            for _attempt in range(20):
                try:
                    client.connect(("127.0.0.1", port))
                    break
                except ConnectionRefusedError:
                    threading.Event().wait(0.01)
            else:
                self.fail("barcode TCP server did not start")
            client.sendall(b"PKG-123456\r\n")
        finally:
            client.close()
        receiver.join(timeout=2.0)

        self.assertEqual(result, ["PKG-123456"])

    def test_nearest_undirected_target_uses_short_path_from_yaw_180(self) -> None:
        target = grasp.nearest_undirected_target_angle_deg(-64.37, 180.0)

        self.assertAlmostEqual(target, 115.63, places=6)
        self.assertAlmostEqual(abs(target - 180.0), 64.37, places=6)

    def setUp(self) -> None:
        self.d_rotation = rpy_xyz_to_matrix(
            np.radians([WAYPOINT_D.rx_deg, WAYPOINT_D.ry_deg, WAYPOINT_D.rz_deg])
        )

    def test_angle_is_undirected_and_uses_shortest_turn(self) -> None:
        self.assertAlmostEqual(wrap_undirected_angle_deg(100.0), -80.0)
        self.assertAlmostEqual(wrap_undirected_angle_deg(-100.0), 80.0)
        self.assertAlmostEqual(wrap_undirected_angle_deg(180.0), 0.0)

    def test_top_plane_ransac_then_svd_rejects_outliers(self) -> None:
        rng = np.random.default_rng(17)
        xy = rng.uniform(-100.0, 100.0, size=(200, 2))
        z = 250.0 + 0.08 * xy[:, 0] - 0.04 * xy[:, 1] + rng.normal(0, 0.15, 200)
        plane_points = np.column_stack([xy, z])
        outliers = rng.uniform([-100, -100, 280], [100, 100, 400], size=(40, 3))
        args = build_parser().parse_args([
            "--top-plane-ransac-threshold-mm", "1",
            "--top-plane-ransac-iterations", "200",
            "--min-top-points", "80",
        ])

        plane = grasp.fit_top_plane_ransac_svd(np.vstack([plane_points, outliers]), args)

        self.assertIsNotNone(plane)
        self.assertGreaterEqual(len(plane.inlier_indices), 195)
        self.assertLess(len(plane.inlier_indices), 210)
        expected = np.array([-0.08, 0.04, 1.0])
        expected /= np.linalg.norm(expected)
        self.assertGreater(abs(float(plane.normal @ expected)), 0.999)

    def test_default_physical_suction_port_mapping(self) -> None:
        args = build_parser().parse_args([])
        cups = {cup.name: cup.do_port for cup in grasp.suction_cup_specs(args)}
        self.assertEqual(
            cups,
            {"primary": 6, "secondary": 5, "third": 4, "fourth": 3},
        )

    def test_all_off_includes_disabled_physical_cups(self) -> None:
        args = build_parser().parse_args([
            "--disable-third-suction",
            "--disable-fourth-suction",
        ])
        robot = object()
        with patch.object(grasp, "set_suction_output") as set_output:
            grasp.set_all_suction_outputs(robot, args, False)
        self.assertEqual(
            [call.kwargs["do_port"] for call in set_output.call_args_list],
            [6, 5, 4, 3],
        )
        self.assertTrue(all(call.args[2] is False for call in set_output.call_args_list))

    def test_default_configuration_includes_perpendicular_third_cup(self) -> None:
        args = build_parser().parse_args([])
        cups = grasp.suction_cup_specs(args)
        third = next(cup for cup in cups if cup.name == "third")

        self.assertEqual(third.do_port, 4)
        np.testing.assert_allclose(third.offset_tool_mm, [102.5, 0.0, 113.0])
        # The virtual cup contact direction (-Z) is main-tool +X.
        np.testing.assert_allclose(
            third.rotation_tool_from_cup @ np.asarray([0.0, 0.0, -1.0]),
            [1.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_perpendicular_third_cup_waypoint_preserves_center_and_orientation(self) -> None:
        args = build_parser().parse_args([])
        third = next(cup for cup in grasp.suction_cup_specs(args) if cup.name == "third")
        commanded_tcp = waypoint_for_selected_cup(WAYPOINT_D, third)

        center = grasp.cup_center_at_tcp_waypoint(commanded_tcp, third)
        np.testing.assert_allclose(
            center,
            [WAYPOINT_D.x_mm, WAYPOINT_D.y_mm, WAYPOINT_D.z_mm],
            atol=1e-8,
        )
        tcp_rotation = rpy_xyz_to_matrix(
            np.radians(
                [commanded_tcp.rx_deg, commanded_tcp.ry_deg, commanded_tcp.rz_deg]
            )
        )
        actual_cup_rotation = grasp.cup_rotation_at_tcp_rotation(tcp_rotation, third)
        np.testing.assert_allclose(actual_cup_rotation, self.d_rotation, atol=1e-8)

    def test_every_selected_cup_reaches_functional_c_center_and_orientation(self) -> None:
        args = build_parser().parse_args([])
        expected_center = np.asarray(
            [grasp.WAYPOINT_C.x_mm, grasp.WAYPOINT_C.y_mm, grasp.WAYPOINT_C.z_mm],
            dtype=np.float64,
        )
        expected_rotation = rpy_xyz_to_matrix(
            np.radians(
                [
                    grasp.WAYPOINT_C.rx_deg,
                    grasp.WAYPOINT_C.ry_deg,
                    grasp.WAYPOINT_C.rz_deg,
                ]
            )
        )

        for cup in grasp.suction_cup_specs(args):
            with self.subTest(cup=cup.name):
                commanded_tcp = waypoint_for_selected_cup(grasp.WAYPOINT_C, cup)
                np.testing.assert_allclose(
                    grasp.cup_center_at_tcp_waypoint(commanded_tcp, cup),
                    expected_center,
                    atol=1e-8,
                )
                commanded_rotation = rpy_xyz_to_matrix(
                    np.radians(
                        [
                            commanded_tcp.rx_deg,
                            commanded_tcp.ry_deg,
                            commanded_tcp.rz_deg,
                        ]
                    )
                )
                np.testing.assert_allclose(
                    grasp.cup_rotation_at_tcp_rotation(commanded_rotation, cup),
                    expected_rotation,
                    atol=1e-8,
                )

    def test_default_configuration_includes_perpendicular_fourth_cup(self) -> None:
        args = build_parser().parse_args([])
        cups = grasp.suction_cup_specs(args)
        third = next(cup for cup in cups if cup.name == "third")
        fourth = next(cup for cup in cups if cup.name == "fourth")

        self.assertEqual(fourth.do_port, 3)
        np.testing.assert_allclose(fourth.offset_tool_mm, [102.5, -245.0, 113.0])
        np.testing.assert_allclose(
            fourth.offset_tool_mm - third.offset_tool_mm,
            [0.0, -245.0, 0.0],
        )
        np.testing.assert_allclose(
            fourth.rotation_tool_from_cup @ np.asarray([0.0, 0.0, -1.0]),
            [1.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_pickup_planning_can_be_restricted_to_third_and_fourth_cups(self) -> None:
        args = build_parser().parse_args(["--force-suction-cups", "3", "4"])

        selected = grasp.selected_suction_cup_specs(args)
        all_outputs = grasp.suction_cup_specs(args)

        self.assertEqual([cup.name for cup in selected], ["third", "fourth"])
        self.assertEqual(
            [cup.name for cup in all_outputs],
            ["primary", "secondary", "third", "fourth"],
        )

    def test_suction_rois_restrict_left_right_and_leave_middle_free(self) -> None:
        args = build_parser().parse_args([])
        roi_config = grasp.RoiConfig(
            overall_polygon=None,
            support_polygons={},
            exclude_polygons=[],
            suction_zone_polygons={
                "left": np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]]),
                "right": np.asarray([[200, 0], [299, 0], [299, 99], [200, 99]]),
            },
        )

        def candidate_at(x: int) -> ClusterCandidate:
            return ClusterCandidate(
                index=1,
                center_pixel=(x, 50),
                hull_pixels=np.asarray([[x, 50], [x + 1, 50], [x, 51]]),
                support_region_id="floor",
                support_region_name="Floor",
                point_camera_mm=np.zeros(3),
                point_base_mm=np.zeros(3),
                normal_camera=np.asarray([0.0, 0.0, 1.0]),
                normal_base=np.asarray([0.0, 0.0, 1.0]),
                height_mm=10.0,
                flatness_mm=1.0,
                point_count=3,
            )

        left, left_zone = grasp.suction_cups_for_candidate(candidate_at(50), args, roi_config)
        middle, middle_zone = grasp.suction_cups_for_candidate(candidate_at(150), args, roi_config)
        right, right_zone = grasp.suction_cups_for_candidate(candidate_at(250), args, roi_config)

        self.assertEqual(left_zone, "left")
        self.assertEqual([cup.name for cup in left], ["primary", "third"])
        self.assertIsNone(middle_zone)
        self.assertEqual([cup.name for cup in middle], ["primary", "secondary", "third", "fourth"])
        self.assertEqual(right_zone, "right")
        self.assertEqual([cup.name for cup in right], ["secondary", "fourth"])

    def test_perpendicular_fourth_cup_waypoint_preserves_center_and_orientation(self) -> None:
        args = build_parser().parse_args([])
        fourth = next(cup for cup in grasp.suction_cup_specs(args) if cup.name == "fourth")
        commanded_tcp = waypoint_for_selected_cup(WAYPOINT_D, fourth)

        center = grasp.cup_center_at_tcp_waypoint(commanded_tcp, fourth)
        np.testing.assert_allclose(
            center,
            [WAYPOINT_D.x_mm, WAYPOINT_D.y_mm, WAYPOINT_D.z_mm],
            atol=1e-8,
        )
        tcp_rotation = rpy_xyz_to_matrix(
            np.radians(
                [commanded_tcp.rx_deg, commanded_tcp.ry_deg, commanded_tcp.rz_deg]
            )
        )
        actual_cup_rotation = grasp.cup_rotation_at_tcp_rotation(tcp_rotation, fourth)
        np.testing.assert_allclose(actual_cup_rotation, self.d_rotation, atol=1e-8)

    def test_perpendicular_cup_loaded_a_star_preserves_taught_tcp_and_selected_face(self) -> None:
        args = build_parser().parse_args([])
        third = next(cup for cup in grasp.suction_cup_specs(args) if cup.name == "third")
        commanded_tcp = grasp.waypoint_with_selected_cup_orientation(
            grasp.WAYPOINT_A_STAR,
            third,
        )

        np.testing.assert_allclose(
            [commanded_tcp.x_mm, commanded_tcp.y_mm, commanded_tcp.z_mm],
            [
                grasp.WAYPOINT_A_STAR.x_mm,
                grasp.WAYPOINT_A_STAR.y_mm,
                grasp.WAYPOINT_A_STAR.z_mm,
            ],
            atol=1e-8,
        )
        tcp_rotation = rpy_xyz_to_matrix(
            np.radians(
                [commanded_tcp.rx_deg, commanded_tcp.ry_deg, commanded_tcp.rz_deg]
            )
        )
        actual_cup_rotation = grasp.cup_rotation_at_tcp_rotation(tcp_rotation, third)
        expected_cup_rotation = rpy_xyz_to_matrix(
            np.radians(
                [
                    grasp.WAYPOINT_A_STAR.rx_deg,
                    grasp.WAYPOINT_A_STAR.ry_deg,
                    grasp.WAYPOINT_A_STAR.rz_deg,
                ]
            )
        )
        np.testing.assert_allclose(actual_cup_rotation, expected_cup_rotation, atol=1e-8)
        # The selected cup's +Z is nearly vertical, so its suction face is
        # nearly parallel to the ground even though the main face is not.
        self.assertGreater(float(actual_cup_rotation[2, 2]), 0.99)

    def test_perpendicular_cups_use_verified_side_loaded_transition(self) -> None:
        args = build_parser().parse_args([])
        cups = grasp.suction_cup_specs(args)
        lifted_rpy = np.asarray([-9.21, 78.13, -15.11])
        for name in ("third", "fourth"):
            cup = next(item for item in cups if item.name == name)
            route = grasp.loaded_clearance_waypoints_for_cup(cup, lifted_rpy)
            self.assertEqual(len(route), 1)
            side_a = route[0]
            np.testing.assert_allclose(
                [side_a.x_mm, side_a.y_mm, side_a.z_mm],
                [-146.465, -1076.584, 656.248],
            )
            np.testing.assert_allclose(
                [side_a.rx_deg, side_a.ry_deg, side_a.rz_deg],
                grasp.SIDE_CUP_FACE_DOWN_RPY_DEG,
            )
            route_rotation = rpy_xyz_to_matrix(
                np.radians([side_a.rx_deg, side_a.ry_deg, side_a.rz_deg])
            )
            np.testing.assert_allclose(
                route_rotation @ np.asarray([1.0, 0.0, 0.0]),
                [-0.01957, -0.04067, -0.99898],
                atol=1e-4,
            )

    def test_parallel_cups_keep_original_loaded_clearance_path(self) -> None:
        args = build_parser().parse_args([])
        cups = grasp.suction_cup_specs(args)
        for name in ("primary", "secondary"):
            cup = next(item for item in cups if item.name == name)
            self.assertEqual(
                grasp.loaded_clearance_waypoints_for_cup(cup),
                (grasp.WAYPOINT_A_STAR, grasp.WAYPOINT_B),
            )

    def test_rpy_unwrap_prevents_270_degree_wrist_command(self) -> None:
        continuous = unwrap_rpy_deg(
            np.asarray([0.0, 0.0, 175.38]),
            np.asarray([-1.917, 2.748, -94.661]),
        )
        self.assertAlmostEqual(continuous[2], -184.62, places=6)
        self.assertLess(abs(continuous[2] - (-94.661)), 91.0)

    def test_gimbal_lock_rpy_components_use_true_rotation_distance(self) -> None:
        # This third-cup pose has a 163-degree Euler component change but only
        # a 107-degree physical orientation change from A*.
        grasp.require_safe_rpy_step(
            np.asarray([-1.917, 2.748, -94.661]),
            np.asarray([161.3, 90.0, 0.0]),
            label="test perpendicular cup",
        )

    def test_side_pose_can_route_to_b_safely_via_a_star(self) -> None:
        side_pose_rpy = np.asarray([9.34, 44.71, 161.14], dtype=np.float64)
        a_star_rpy = np.asarray(
            [
                grasp.WAYPOINT_A_STAR.rx_deg,
                grasp.WAYPOINT_A_STAR.ry_deg,
                grasp.WAYPOINT_A_STAR.rz_deg,
            ]
        )
        b_rpy = np.asarray(
            [grasp.WAYPOINT_B.rx_deg, grasp.WAYPOINT_B.ry_deg, grasp.WAYPOINT_B.rz_deg]
        )

        with self.assertRaisesRegex(RuntimeError, "Unsafe RPY step"):
            grasp.require_safe_rpy_step(side_pose_rpy, b_rpy, label="direct side -> B")
        grasp.require_safe_rpy_step(side_pose_rpy, a_star_rpy, label="side -> A*")
        grasp.require_safe_rpy_step(a_star_rpy, b_rpy, label="A* -> B")

    def test_d_side_pose_can_route_to_a_star_safely_via_b(self) -> None:
        current_rpy = np.asarray([-3.67, -12.96, 77.15], dtype=np.float64)
        a_star_rpy = np.asarray(
            [
                grasp.WAYPOINT_A_STAR.rx_deg,
                grasp.WAYPOINT_A_STAR.ry_deg,
                grasp.WAYPOINT_A_STAR.rz_deg,
            ],
            dtype=np.float64,
        )
        b_rpy = np.asarray(
            [grasp.WAYPOINT_B.rx_deg, grasp.WAYPOINT_B.ry_deg, grasp.WAYPOINT_B.rz_deg],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(RuntimeError, "Unsafe RPY step"):
            grasp.require_safe_rpy_step(current_rpy, a_star_rpy, label="direct current -> A*")
        grasp.require_safe_rpy_step(current_rpy, b_rpy, label="current -> B")
        grasp.require_safe_rpy_step(b_rpy, a_star_rpy, label="B -> A*")

    def test_pose_at_b_is_detected_without_redundant_recovery_route(self) -> None:
        b_xyz = np.asarray(
            [grasp.WAYPOINT_B.x_mm, grasp.WAYPOINT_B.y_mm, grasp.WAYPOINT_B.z_mm]
        )
        b_rpy = np.asarray(
            [grasp.WAYPOINT_B.rx_deg, grasp.WAYPOINT_B.ry_deg, grasp.WAYPOINT_B.rz_deg]
        )
        pose = SimpleNamespace(
            translation_mm=lambda: b_xyz + np.asarray([1.0, -1.0, 0.5]),
            rpy_rad_xyz=np.radians(b_rpy + np.asarray([0.2, -0.2, 0.3])),
        )
        self.assertTrue(grasp.pose_is_at_waypoint(pose, grasp.WAYPOINT_B))

    def test_empty_move_retries_no_ik_without_confdata(self) -> None:
        calls = []

        def move(*args, **kwargs):
            calls.append(kwargs["options"])
            if len(calls) == 1:
                raise RuntimeError(
                    "Robot move rejected before start: {'ec': -50021, "
                    "'message': 'specified conf has no solution'}"
                )
            return SimpleNamespace()

        robot = SimpleNamespace(move_to_pose_mm_deg=move, stop_motion=lambda: None)
        options = grasp.MotionOptions(motion="movej", use_current_conf_data=True)
        grasp.move_pose_with_singularity_fallback(
            robot,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            options,
            allow_clear_confdata_retry=True,
            allow_movej_singularity_retry=False,
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].use_current_conf_data)
        self.assertFalse(calls[1].use_current_conf_data)

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

    def test_default_pickup_uses_fitted_top_plane(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.normal_mode, "top-plane")
        pickup_rpy = compute_grasp_rpy_from_normal_and_fixed_x(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            FIXED_GRASP_X_YAW_DEG,
            "minus-z",
        )
        np.testing.assert_allclose(pickup_rpy[:2], [0.0, 0.0], atol=1e-9)

    def test_tilted_top_plane_aligns_suction_contact_axis(self) -> None:
        normal = np.asarray([0.25, -0.18, 0.951], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        pickup_rpy = compute_grasp_rpy_from_normal_and_fixed_x(
            normal,
            FIXED_GRASP_X_YAW_DEG,
            "minus-z",
        )
        pickup_rotation = rpy_xyz_to_matrix(np.radians(pickup_rpy))

        # For minus-z contact, TCP -Z must point into the surface, opposite
        # the outward fitted top normal. Therefore TCP +Z equals the normal.
        np.testing.assert_allclose(pickup_rotation[:, 2], normal, atol=1e-8)

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

    def test_selected_cup_rotation_can_unwind_at_safe_point(self) -> None:
        primary = SuctionCupSpec(
            name="primary",
            do_port=6,
            offset_tool_mm=np.zeros(3, dtype=np.float64),
        )
        commanded_offsets = []

        def fake_move(physical_waypoint, cup, _robot, _motion_options, **_kwargs):
            reference_rotation = rpy_xyz_to_matrix(
                np.radians(
                    [
                        grasp.WAYPOINT_D_ROTATE_SAFE.rx_deg,
                        grasp.WAYPOINT_D_ROTATE_SAFE.ry_deg,
                        grasp.WAYPOINT_D_ROTATE_SAFE.rz_deg,
                    ]
                )
            )
            waypoint_rotation = rpy_xyz_to_matrix(
                np.radians(
                    [
                        physical_waypoint.rx_deg,
                        physical_waypoint.ry_deg,
                        physical_waypoint.rz_deg,
                    ]
                )
            )
            commanded_offsets.append(
                grasp.rotation_distance_deg(reference_rotation, waypoint_rotation)
            )
            return physical_waypoint, np.asarray(
                [physical_waypoint.x_mm, physical_waypoint.y_mm, physical_waypoint.z_mm]
            )

        with patch.object(
            grasp,
            "move_selected_cup_to_functional_waypoint",
            side_effect=fake_move,
        ):
            rotate_selected_cup_at_functional_waypoint(
                grasp.WAYPOINT_D_ROTATE_SAFE,
                0.0,
                primary,
                object(),
                object(),
                start_local_z_deg=56.0,
                max_step_deg=10.0,
            )

        self.assertEqual(len(commanded_offsets), 6)
        self.assertAlmostEqual(commanded_offsets[-1], 0.0, places=6)
        self.assertTrue(
            all(a > b for a, b in zip(commanded_offsets, commanded_offsets[1:]))
        )

    def test_failed_segmented_rotation_refines_and_keeps_closest_reachable_angle(self) -> None:
        primary = SuctionCupSpec("primary", 6, np.zeros(3))
        commanded_offsets = []
        failed_once = False

        def fake_move(physical_waypoint, _cup, _robot, _options, **kwargs):
            nonlocal failed_once
            reference = rpy_xyz_to_matrix(np.radians([
                grasp.WAYPOINT_D_ROTATE_SAFE.rx_deg,
                grasp.WAYPOINT_D_ROTATE_SAFE.ry_deg,
                grasp.WAYPOINT_D_ROTATE_SAFE.rz_deg,
            ]))
            actual = rpy_xyz_to_matrix(np.radians([
                physical_waypoint.rx_deg, physical_waypoint.ry_deg, physical_waypoint.rz_deg,
            ]))
            signed = np.degrees(np.arctan2((reference.T @ actual)[1, 0], (reference.T @ actual)[0, 0]))
            commanded_offsets.append(float(signed))
            self.assertFalse(kwargs["allow_current_conf_movej_fallback"])
            if len(commanded_offsets) == 6 and not failed_once:
                failed_once = True
                raise RuntimeError("-50102 path singularity")
            return physical_waypoint, np.array([physical_waypoint.x_mm, physical_waypoint.y_mm, physical_waypoint.z_mm])

        with patch.object(grasp, "move_selected_cup_to_functional_waypoint", side_effect=fake_move):
            reached, _center = rotate_selected_cup_at_functional_waypoint(
                grasp.WAYPOINT_D_ROTATE_SAFE,
                80.0,
                primary,
                object(),
                object(),
                max_step_deg=10.0,
            )

        np.testing.assert_allclose(commanded_offsets[:6], [10, 20, 30, 40, 50, 60], atol=1e-6)
        self.assertAlmostEqual(commanded_offsets[6], 50.0, places=6)
        reached_rotation = rpy_xyz_to_matrix(
            np.radians([reached.rx_deg, reached.ry_deg, reached.rz_deg])
        )
        reference_rotation = rpy_xyz_to_matrix(
            np.radians([
                grasp.WAYPOINT_D_ROTATE_SAFE.rx_deg,
                grasp.WAYPOINT_D_ROTATE_SAFE.ry_deg,
                grasp.WAYPOINT_D_ROTATE_SAFE.rz_deg,
            ])
        )
        reached_delta = np.degrees(
            np.arctan2(
                (reference_rotation.T @ reached_rotation)[1, 0],
                (reference_rotation.T @ reached_rotation)[0, 0],
            )
        )
        self.assertGreater(reached_delta, 59.0)
        self.assertLess(reached_delta, 60.0)

    def test_primary_and_secondary_placement_share_the_same_d_center(self) -> None:
        primary = SuctionCupSpec(
            name="primary",
            do_port=6,
            offset_tool_mm=np.zeros(3, dtype=np.float64),
        )
        secondary = SuctionCupSpec(
            name="secondary",
            do_port=5,
            offset_tool_mm=np.asarray([0.0, -245.0, 0.0], dtype=np.float64),
        )
        aligned_d = rotate_waypoint_about_local_z(WAYPOINT_D, -50.49)
        primary_tcp, primary_center = placement_tcp_waypoint_for_selected_cup(aligned_d, primary)
        secondary_tcp, secondary_center = placement_tcp_waypoint_for_selected_cup(aligned_d, secondary)

        np.testing.assert_allclose(primary_center, secondary_center, atol=1e-8)
        np.testing.assert_allclose(
            secondary_center,
            [aligned_d.x_mm, aligned_d.y_mm, aligned_d.z_mm],
            atol=1e-8,
        )
        self.assertNotAlmostEqual(primary_tcp.x_mm, secondary_tcp.x_mm, places=6)
        self.assertNotAlmostEqual(primary_tcp.y_mm, secondary_tcp.y_mm, places=6)

    def test_barcode_side_view_keeps_c_center_and_uses_taught_relative_rotation(self) -> None:
        side_view = barcode_side_view_waypoint(WAYPOINT_D)
        self.assertEqual(
            (side_view.x_mm, side_view.y_mm, side_view.z_mm),
            (WAYPOINT_D.x_mm, WAYPOINT_D.y_mm, WAYPOINT_D.z_mm),
        )
        reference_rotation = rpy_xyz_to_matrix(
            np.radians([WAYPOINT_D.rx_deg, WAYPOINT_D.ry_deg, WAYPOINT_D.rz_deg])
        )
        side_rotation = rpy_xyz_to_matrix(
            np.radians([side_view.rx_deg, side_view.ry_deg, side_view.rz_deg])
        )
        np.testing.assert_allclose(
            reference_rotation.T @ side_rotation,
            grasp.C_BARCODE_SIDE_VIEW_ROTATION,
            atol=1e-8,
        )
        side_angle_deg = grasp.rotation_distance_deg(reference_rotation, side_rotation)
        self.assertGreater(side_angle_deg, 75.0)

    def test_primary_and_secondary_barcode_poses_match_verified_rotation_test(self) -> None:
        initial = grasp.WAYPOINT_C_BARCODE
        side = grasp.WAYPOINT_C_BARCODE_SIDE
        np.testing.assert_allclose(
            [initial.x_mm, initial.y_mm, initial.z_mm, initial.rx_deg, initial.ry_deg, initial.rz_deg],
            [627.094, 311.420, 470.689, -2.136, 3.046, -3.744],
        )
        np.testing.assert_allclose(
            [side.x_mm, side.y_mm, side.z_mm, side.rx_deg, side.ry_deg, side.rz_deg],
            [627.067, 311.427, 470.629, -0.736, 3.143, 75.357],
        )
        initial_rotation = rpy_xyz_to_matrix(
            np.radians([initial.rx_deg, initial.ry_deg, initial.rz_deg])
        )
        side_rotation = rpy_xyz_to_matrix(
            np.radians([side.rx_deg, side.ry_deg, side.rz_deg])
        )
        self.assertGreater(grasp.rotation_distance_deg(initial_rotation, side_rotation), 75.0)

    def test_all_c_routes_share_the_verified_rotation_test_pose(self) -> None:
        self.assertIs(grasp.WAYPOINT_C_BARCODE, grasp.WAYPOINT_C)
        np.testing.assert_allclose(
            [
                grasp.WAYPOINT_C.x_mm,
                grasp.WAYPOINT_C.y_mm,
                grasp.WAYPOINT_C.z_mm,
                grasp.WAYPOINT_C.rx_deg,
                grasp.WAYPOINT_C.ry_deg,
                grasp.WAYPOINT_C.rz_deg,
            ],
            [627.094, 311.420, 470.689, -2.136, 3.046, -3.744],
        )


if __name__ == "__main__":
    unittest.main()
