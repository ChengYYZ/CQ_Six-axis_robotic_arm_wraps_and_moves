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
    LatestRGBDFrameBuffer,
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
    pickup_alignment_allows_direct_c_to_d,
    placement_aligned_pickup_rpy_candidates,
    placement_tcp_waypoint_for_selected_cup,
    progressive_placement_fallback_angles_deg,
    move_empty_rotation_safe_to_b,
    receive_barcode_from_tcp_client,
    reverse_placement_trial_angles_deg,
    retreat_to_b_when_no_packages,
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
    @staticmethod
    def _candidate(class_name: str) -> ClusterCandidate:
        return ClusterCandidate(
            index=1,
            center_pixel=(10, 10),
            hull_pixels=np.asarray([[0, 0], [20, 0], [20, 20], [0, 20]]),
            support_region_id="bin",
            support_region_name="bin",
            point_camera_mm=np.zeros(3),
            point_base_mm=np.zeros(3),
            normal_camera=np.asarray([0.0, 0.0, -1.0]),
            normal_base=np.asarray([0.0, 0.0, 1.0]),
            height_mm=20.0,
            flatness_mm=1.0,
            point_count=100,
            class_name=class_name,
        )

    def test_soft_package_skips_long_edge_pickup_and_d_rotation(self) -> None:
        candidate = self._candidate("soft_parcel")
        current_pose = SimpleNamespace(rpy_rad_xyz=np.zeros(3))
        args = build_parser().parse_args([])

        self.assertEqual(
            placement_aligned_pickup_rpy_candidates(candidate, current_pose, args),
            [],
        )
        safe, placed, rotation_deg, error_deg = grasp.aligned_placement_waypoints(
            candidate, np.zeros(3)
        )
        self.assertIs(safe, grasp.WAYPOINT_D_ROTATE_SAFE)
        self.assertIs(placed, grasp.WAYPOINT_D)
        self.assertEqual((rotation_deg, error_deg), (0.0, 0.0))

    def test_loaded_b_and_c_keep_restored_a_star_configuration_locked(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.settled_b_pose = SimpleNamespace(
                    rpy_deg_xyz=lambda: np.asarray(
                        [grasp.WAYPOINT_B.rx_deg, grasp.WAYPOINT_B.ry_deg, grasp.WAYPOINT_B.rz_deg]
                    ),
                    conf_data=[1.0, 2.0, 3.0, 4.0],
                )

            def read_current_pose(self):
                return self.settled_b_pose

        robot = FakeRobot()
        with patch.object(grasp, "move_pose_with_singularity_fallback") as move:
            grasp.move_loaded_through_b_to_c(
                robot,
                grasp.MotionOptions(),
                np.asarray(
                    [
                        grasp.WAYPOINT_A_STAR.rx_deg,
                        grasp.WAYPOINT_A_STAR.ry_deg,
                        grasp.WAYPOINT_A_STAR.rz_deg,
                    ]
                ),
                pass_through_zone_mm=30.0,
            )

        self.assertEqual(move.call_count, 2)
        b_call, c_call = move.call_args_list
        self.assertEqual(b_call.args[1:4], (grasp.WAYPOINT_B.x_mm, grasp.WAYPOINT_B.y_mm, grasp.WAYPOINT_B.z_mm))
        self.assertEqual(c_call.args[1:4], (grasp.WAYPOINT_C_BARCODE.x_mm, grasp.WAYPOINT_C_BARCODE.y_mm, grasp.WAYPOINT_C_BARCODE.z_mm))
        self.assertEqual(b_call.args[7].motion, "movej")
        self.assertTrue(b_call.args[7].use_current_conf_data)
        self.assertEqual(b_call.args[7].zone_mm, 0.0)
        self.assertEqual(c_call.args[7].motion, "movej")
        self.assertTrue(c_call.args[7].use_current_conf_data)
        self.assertEqual(c_call.args[7].zone_mm, 0.0)
        self.assertFalse(c_call.kwargs["allow_clear_confdata_retry"])

    def test_empty_d_return_blends_through_c_and_b_then_stops_at_a_star(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.path = None

            def read_current_pose(self):
                return SimpleNamespace(
                    translation_mm=lambda: np.asarray(
                        [grasp.WAYPOINT_D.x_mm, grasp.WAYPOINT_D.y_mm, grasp.WAYPOINT_D.z_mm]
                    ),
                    rpy_deg_xyz=lambda: np.asarray([-0.35, 3.12, 147.66]),
                )

            def move_path_mm_deg(self, path, options):
                self.path = path
                return SimpleNamespace(rpy_deg_xyz=lambda: np.asarray(path[-1][3:6]))

        robot = FakeRobot()
        grasp.return_empty_from_d_through_c_b(
            robot,
            grasp.MotionOptions(),
            pass_through_zone_mm=30.0,
        )
        self.assertEqual(len(robot.path), 4)
        self.assertEqual(robot.path[0][6:], ("movej", 30.0))
        self.assertEqual(robot.path[1][6:], ("movej", 30.0))
        self.assertEqual(robot.path[2][6:], ("movej", 30.0))
        self.assertEqual(robot.path[3][6:], ("movej", 0.0))
        self.assertEqual(robot.path[1][:3], (
            grasp.WAYPOINT_C.x_mm, grasp.WAYPOINT_C.y_mm, grasp.WAYPOINT_C.z_mm
        ))
        self.assertEqual(robot.path[2][:3], (
            grasp.WAYPOINT_B.x_mm, grasp.WAYPOINT_B.y_mm, grasp.WAYPOINT_B.z_mm
        ))
        self.assertEqual(robot.path[3][:3], (
            grasp.WAYPOINT_A_STAR.x_mm,
            grasp.WAYPOINT_A_STAR.y_mm,
            grasp.WAYPOINT_A_STAR.z_mm,
        ))

    def test_latest_rgbd_buffer_returns_distinct_new_sequences(self) -> None:
        class FakeCamera:
            def __init__(self) -> None:
                self.value = 0

            def get_frames(self, _timeout_ms: int):
                threading.Event().wait(0.005)
                self.value += 1
                frame = np.full((2, 2), self.value, dtype=np.uint16)
                return frame, frame, frame.astype(np.uint8)

        buffer = LatestRGBDFrameBuffer(FakeCamera(), 20)
        buffer.start()
        try:
            first = buffer.get_latest(500)
            self.assertIsNotNone(first)
            first_sequence, _first_frames = first
            second = buffer.get_latest(500, after_sequence=first_sequence)
            self.assertIsNotNone(second)
            second_sequence, _second_frames = second
            self.assertGreater(second_sequence, first_sequence)
        finally:
            buffer.stop()

    def test_next_scene_suction_plans_are_precomputed_from_a_star(self) -> None:
        candidate = self._candidate("soft_parcel")
        analysis = grasp.AnalysisResult(
            color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_mm=np.zeros((2, 2), dtype=np.uint16),
            depth_display=np.zeros((2, 2), dtype=np.uint8),
            base_plane=None,
            candidates=[candidate],
            notes=[],
        )
        args = build_parser().parse_args([])
        sentinel_plan = object()

        with patch.object(
            grasp,
            "compute_approach_geometry_from_pose",
            return_value=(
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([1.0, 2.0, 2.0]),
                np.zeros(3),
                "test",
                [("test", np.zeros(3))],
            ),
        ) as compute, patch.object(
            grasp,
            "build_suction_approach_plans",
            return_value=[sentinel_plan],
        ) as build:
            plans = grasp.precompute_suction_plans_from_a_star(analysis, args, None)

        planning_pose = compute.call_args.args[1]
        np.testing.assert_allclose(
            planning_pose.translation_mm(),
            [grasp.WAYPOINT_A_STAR.x_mm, grasp.WAYPOINT_A_STAR.y_mm, grasp.WAYPOINT_A_STAR.z_mm],
        )
        np.testing.assert_allclose(
            build.call_args.args[2], planning_pose.translation_mm()
        )
        self.assertEqual(plans[candidate.index], [sentinel_plan])

    def test_no_detected_package_moves_empty_tool_from_a_star_to_b(self) -> None:
        a_star_pose = SimpleNamespace(
            translation_mm=lambda: np.asarray(
                [grasp.WAYPOINT_A_STAR.x_mm, grasp.WAYPOINT_A_STAR.y_mm, grasp.WAYPOINT_A_STAR.z_mm]
            ),
            rpy_rad_xyz=np.radians(
                [grasp.WAYPOINT_A_STAR.rx_deg, grasp.WAYPOINT_A_STAR.ry_deg, grasp.WAYPOINT_A_STAR.rz_deg]
            ),
        )
        robot = SimpleNamespace(read_current_pose=lambda: a_star_pose)

        with patch.object(grasp, "move_empty_to_waypoint_segmented") as move:
            handled = retreat_to_b_when_no_packages(
                [], robot, grasp.MotionOptions(), dry_run=False
            )

        self.assertTrue(handled)
        move.assert_called_once_with(grasp.WAYPOINT_B, robot, unittest.mock.ANY)

    def test_detected_package_does_not_trigger_camera_clear_retreat(self) -> None:
        robot = SimpleNamespace(read_current_pose=lambda: None)

        with patch.object(grasp, "move_empty_to_waypoint_segmented") as move:
            handled = retreat_to_b_when_no_packages(
                [self._candidate("parcel_box")], robot, grasp.MotionOptions(), dry_run=False
            )

        self.assertFalse(handled)
        move.assert_not_called()

    def test_demo_workspace_extends_to_negative_550_mm(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.workspace_x_min_mm, -550.0)

    def test_d_safe_rotation_has_independent_speed(self) -> None:
        defaults = build_parser().parse_args([])
        configured = build_parser().parse_args(
            ["--robot-speed-mm-s", "400", "--d-safe-rotation-speed-mm-s", "60"]
        )
        self.assertEqual(defaults.d_safe_rotation_speed_mm_s, 80.0)
        self.assertEqual(configured.robot_speed_mm_s, 400.0)
        self.assertEqual(configured.d_safe_rotation_speed_mm_s, 60.0)

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

    def test_aligned_pickup_skips_loaded_rotation_safe_waypoint(self) -> None:
        self.assertTrue(pickup_alignment_allows_direct_c_to_d(0.0, 5.0))
        self.assertTrue(pickup_alignment_allows_direct_c_to_d(-4.99, 5.0))
        self.assertTrue(pickup_alignment_allows_direct_c_to_d(175.0, 5.0))
        self.assertFalse(pickup_alignment_allows_direct_c_to_d(5.01, 5.0))

    def test_placement_defaults_prefer_alignment_without_blocking_safe_rotation(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(list(args.verified_placement_angles_deg), [0.0])
        self.assertEqual(args.max_placement_alignment_error_deg, 5.0)
        self.assertTrue(args.allow_degraded_placement)
        strict_args = build_parser().parse_args(["--require-aligned-placement"])
        self.assertFalse(strict_args.allow_degraded_placement)
        self.assertEqual(args.max_controller_plan_attempts, 8)
        self.assertEqual(args.max_auto_conf_singularity_attempts, 2)
        self.assertFalse(args.disable_auto_conf_singularity_recovery)
        self.assertEqual(
            list(args.singularity_recovery_yaw_offsets_deg),
            [2.0, -2.0, 4.0, -4.0],
        )
        self.assertEqual(args.auto_cycle_settle_s, 0.0)

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

    def test_progressive_d_search_tries_each_step_before_verified_fallback(self) -> None:
        angles = progressive_placement_fallback_angles_deg(
            requested_deg=86.31,
            current_deg=86.31,
            verified_angles_deg=[0.0],
            step_deg=10.0,
        )

        self.assertAlmostEqual(angles[0], 76.31)
        self.assertAlmostEqual(angles[1], 66.31)
        self.assertAlmostEqual(angles[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(angles, angles[1:])))

    def test_progressive_d_search_orders_verified_targets_by_alignment(self) -> None:
        angles = progressive_placement_fallback_angles_deg(
            requested_deg=42.0,
            current_deg=42.0,
            verified_angles_deg=[0.0, 30.0],
            step_deg=10.0,
        )

        self.assertEqual(angles[:2], [32.0, 30.0])
        self.assertEqual(angles[-1], 0.0)

    def test_reverse_d_search_stops_at_45_degrees_then_tries_original(self) -> None:
        angles = reverse_placement_trial_angles_deg(
            requested_deg=80.0,
            current_deg=80.0,
            original_deg=0.0,
            step_deg=10.0,
            max_alignment_error_deg=45.0,
        )
        self.assertEqual(angles, [70.0, 60.0, 50.0, 40.0, 0.0])

    def test_reverse_d_search_uses_actual_high_point_entry_angle(self) -> None:
        angles = reverse_placement_trial_angles_deg(
            requested_deg=-55.0,
            current_deg=-55.0,
            original_deg=-5.0,
            step_deg=10.0,
            max_alignment_error_deg=45.0,
        )
        self.assertEqual(angles, [-45.0, -35.0, -25.0, -15.0, -5.0])

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

    def test_restart_recovery_accepts_existing_d_safe_high_pose(self) -> None:
        high_xyz = np.asarray([34.4, 702.9, 524.8])
        high_rpy = np.asarray([-0.02, 2.45, 141.29])
        robot = SimpleNamespace(
            read_current_pose=lambda: SimpleNamespace(
                translation_mm=lambda: high_xyz.copy(),
                rpy_deg_xyz=lambda: high_rpy.copy(),
            )
        )
        rotation_xyz: list[np.ndarray] = []
        reached_waypoints: list[object] = []

        with patch.object(
            grasp,
            "move_pose_with_singularity_fallback",
            side_effect=lambda _robot, x, y, z, *_args, **_kwargs: rotation_xyz.append(
                np.asarray([x, y, z])
            ),
        ), patch.object(
            grasp,
            "move_waypoint",
            side_effect=lambda waypoint, *_args, **_kwargs: reached_waypoints.append(waypoint),
        ):
            grasp.recover_empty_tool_from_d_to_b(robot, grasp.MotionOptions())

        self.assertTrue(rotation_xyz)
        for xyz in rotation_xyz:
            np.testing.assert_allclose(xyz, high_xyz)
        self.assertEqual(reached_waypoints, [grasp.WAYPOINT_B])

    def test_empty_d_return_only_lifts_at_rotation_safe_point(self) -> None:
        cup = SuctionCupSpec("primary", 6, np.zeros(3))
        calls: list[tuple[str, float]] = []

        with patch.object(
            grasp,
            "move_selected_cup_to_functional_waypoint",
            side_effect=lambda waypoint, *_args, **_kwargs: calls.append(
                ("lift", waypoint.rz_deg)
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
            ["lift", "listen"],
        )

    def test_empty_tool_normalizes_while_moving_to_b(self) -> None:
        pose = SimpleNamespace(
            translation_mm=lambda: np.asarray([-17.6, 577.4, 526.3], dtype=np.float64),
            rpy_deg_xyz=lambda: np.asarray([-0.02, 2.45, 141.37], dtype=np.float64)
        )
        robot = SimpleNamespace(read_current_pose=lambda: pose)
        commands: list[tuple[float, ...]] = []

        with patch.object(
            grasp,
            "move_pose_with_singularity_fallback",
            side_effect=lambda _robot, *values, **_kwargs: commands.append(values),
        ):
            move_empty_rotation_safe_to_b(robot, grasp.MotionOptions())

        self.assertEqual(len(commands), 3)
        self.assertAlmostEqual(commands[-1][0], grasp.WAYPOINT_B.x_mm)
        self.assertAlmostEqual(commands[-1][1], grasp.WAYPOINT_B.y_mm)
        self.assertAlmostEqual(commands[-1][2], grasp.WAYPOINT_B.z_mm)
        self.assertAlmostEqual(commands[-1][5], grasp.WAYPOINT_B.rz_deg)

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
        self.assertEqual(args.place_dwell_s, 0.0)
        self.assertEqual(args.barcode_reader_timeout_s, 3.0)

    def test_waybill_capture_defaults_reduce_fixed_robot_dwell(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.waybill_start_delay_s, 0.0)
        self.assertEqual(args.waybill_c_settle_s, 0.4)
        self.assertEqual(args.waybill_post_c_capture_s, 1.5)
        self.assertEqual(args.waybill_c_dwell_s, 2.0)

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

    def test_disabled_pickup_cups_remain_physical_collision_bodies(self) -> None:
        args = build_parser().parse_args([
            "--disable-third-suction",
            "--disable-fourth-suction",
        ])

        self.assertEqual(
            [cup.name for cup in grasp.suction_cup_specs(args)],
            ["primary", "secondary"],
        )
        self.assertEqual(
            [cup.name for cup in grasp.physical_suction_cup_specs(args)],
            ["primary", "secondary", "third", "fourth"],
        )

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

    def test_high_clearance_recovery_uses_auto_conf_only_before_movel(self) -> None:
        a_star_xyz = np.asarray(
            [grasp.WAYPOINT_A_STAR.x_mm, grasp.WAYPOINT_A_STAR.y_mm, grasp.WAYPOINT_A_STAR.z_mm]
        )
        a_star_rpy = np.asarray(
            [grasp.WAYPOINT_A_STAR.rx_deg, grasp.WAYPOINT_A_STAR.ry_deg, grasp.WAYPOINT_A_STAR.rz_deg]
        )

        def pose(xyz, rpy, conf, elbow):
            return SimpleNamespace(
                translation_mm=lambda: np.asarray(xyz, dtype=np.float64),
                rpy_deg_xyz=lambda: np.asarray(rpy, dtype=np.float64),
                rpy_rad_xyz=np.radians(rpy),
                conf_data=list(conf),
                elbow=float(elbow),
            )

        a_star_pose = pose(a_star_xyz, a_star_rpy, [0, 0, 0, 0], 0.0)
        reached_pose = pose([180.0, -760.0, 200.0], [2.0, -4.0, -88.0], [1, 0, 0, 0], 1.0)
        move_options = []
        conf_policy = []

        robot = SimpleNamespace(
            read_current_pose=lambda: a_star_pose,
            read_current_joints_rad=lambda: np.radians(
                [6.277, 20.0, 40.0, -100.0, 80.0, 10.0]
            ),
            set_conf_data_forced=lambda forced: conf_policy.append(forced),
        )

        def fake_move(*_args, **kwargs):
            move_options.append(kwargs["motion_options"] if "motion_options" in kwargs else _args[7])
            return reached_pose

        with patch.object(grasp, "move_pose_with_singularity_fallback", side_effect=fake_move):
            result = grasp.move_empty_approach_to_a(
                robot,
                np.asarray([180.0, -760.0, 200.0]),
                np.asarray([2.0, -4.0, -88.0]),
                grasp.MotionOptions(motion="movej", use_current_conf_data=True),
                use_auto_conf=True,
            )

        self.assertEqual(conf_policy, [False, True])
        self.assertEqual(len(move_options), 1)
        self.assertEqual(move_options[0].motion, "movej")
        self.assertFalse(move_options[0].use_current_conf_data)
        np.testing.assert_allclose(
            np.degrees(result.a_star_joints_rad),
            [6.277, 20.0, 40.0, -100.0, 80.0, 10.0],
            atol=1e-9,
        )

    def test_a_star_joint_reference_is_session_stable_and_restored_with_moveabsj(self) -> None:
        original = np.radians([6.0, 20.0, 40.0, -100.0, 80.0, 10.0])
        wrong_branch = np.radians([6.0, 69.0, 69.0, -191.0, 148.0, 86.0])
        state = {"joints": original.copy()}
        restores = []

        def restore(joints, options):
            restores.append((np.asarray(joints).copy(), options))
            state["joints"] = np.asarray(joints).copy()

        robot = SimpleNamespace(
            read_current_joints_rad=lambda: state["joints"].copy(),
            move_to_joint_positions_rad=restore,
        )
        options = grasp.MotionOptions(speed_mm_s=150.0)

        first = grasp.capture_or_restore_a_star_joint_reference(robot, options)
        state["joints"] = wrong_branch.copy()
        second = grasp.capture_or_restore_a_star_joint_reference(robot, options)

        np.testing.assert_allclose(first, original)
        np.testing.assert_allclose(second, original)
        self.assertEqual(len(restores), 1)
        np.testing.assert_allclose(restores[0][0], original)
        self.assertEqual(restores[0][1].speed_mm_s, 150.0)
        self.assertEqual(restores[0][1].zone_mm, 0.0)

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

    def test_loaded_movej_at_c_keeps_current_confdata_and_cannot_clear_it(self) -> None:
        current_pose = SimpleNamespace(
            rpy_deg_xyz=lambda: np.asarray([-0.296, 1.428, -6.094], dtype=np.float64)
        )
        robot = SimpleNamespace(read_current_pose=lambda: current_pose)
        motion_options = grasp.MotionOptions(motion="movej", use_current_conf_data=True)

        with patch.object(grasp, "move_pose_with_singularity_fallback") as move:
            grasp.move_loaded_transfer_with_singularity_fallback(
                grasp.WAYPOINT_C_BARCODE_SIDE,
                robot,
                motion_options,
            )

        commanded_options = move.call_args.args[7]
        self.assertEqual(commanded_options.motion, "movej")
        self.assertTrue(commanded_options.use_current_conf_data)
        self.assertFalse(move.call_args.kwargs["allow_clear_confdata_retry"])
        self.assertFalse(move.call_args.kwargs["allow_movej_singularity_retry"])

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
        self.assertEqual(len(candidates), 10)
        self.assertNotIn("_singularity_recovery_", candidates[0][0])
        self.assertNotIn("_singularity_recovery_", candidates[1][0])
        self.assertTrue(
            all("_singularity_recovery_" in mode for mode, _rpy in candidates[2:])
        )

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

    def test_undirected_long_edge_keeps_near_directed_tcp_branch_first(self) -> None:
        candidate = self._candidate("parcel_box")
        long_yaw_deg = 178.7
        long_axis = np.asarray(
            [np.cos(np.radians(long_yaw_deg)), np.sin(np.radians(long_yaw_deg)), 0.0]
        )
        candidate.short_axis_base = np.asarray(
            [long_axis[1], -long_axis[0], 0.0], dtype=np.float64
        )
        current_pose = SimpleNamespace(
            rpy_rad_xyz=np.radians([0.0, 0.0, -94.661])
        )
        args = build_parser().parse_args([])
        args.tool_contact_axis = "minus-z"

        candidates = placement_aligned_pickup_rpy_candidates(
            candidate, current_pose, args
        )
        self.assertEqual(len(candidates), 10)
        current_rotation = rpy_xyz_to_matrix(current_pose.rpy_rad_xyz)
        rotation_distances = [
            grasp.rotation_distance_deg(
                current_rotation,
                rpy_xyz_to_matrix(np.radians(candidate_rpy_deg)),
            )
            for _mode, candidate_rpy_deg in candidates
        ]

        self.assertLess(rotation_distances[0], 10.0)
        self.assertGreater(rotation_distances[1], 170.0)
        exact_z = rpy_xyz_to_matrix(np.radians(candidates[0][1]))[:, 2]
        for mode, recovery_rpy_deg in candidates[2:]:
            self.assertIn("_singularity_recovery_", mode)
            recovery_z = rpy_xyz_to_matrix(np.radians(recovery_rpy_deg))[:, 2]
            np.testing.assert_allclose(recovery_z, exact_z, atol=1e-8)

    def test_long_edge_planning_rejects_opposite_tcp_branch_over_rotation_limit(self) -> None:
        candidate = self._candidate("parcel_box")
        long_yaw_deg = 178.7
        long_axis = np.asarray(
            [np.cos(np.radians(long_yaw_deg)), np.sin(np.radians(long_yaw_deg)), 0.0]
        )
        candidate.short_axis_base = np.asarray(
            [long_axis[1], -long_axis[0], 0.0], dtype=np.float64
        )
        candidate.point_base_mm = np.asarray([0.0, -700.0, 20.0])
        args = build_parser().parse_args([])
        args.tool_contact_axis = "minus-z"
        args.force_suction_cups = ["1", "2"]
        args.unused_cup_min_clearance_mm = 0.0
        args.suction_cup_collision_radius_mm = 0.0
        current_rpy_deg = np.asarray([0.0, 0.0, -94.661])
        current_pose = SimpleNamespace(rpy_rad_xyz=np.radians(current_rpy_deg))
        rpy_candidates = placement_aligned_pickup_rpy_candidates(
            candidate, current_pose, args
        )

        plans = grasp.build_suction_approach_plans(
            candidate,
            [candidate],
            np.asarray(
                [grasp.WAYPOINT_A_STAR.x_mm, grasp.WAYPOINT_A_STAR.y_mm, grasp.WAYPOINT_A_STAR.z_mm]
            ),
            current_rpy_deg,
            np.asarray([0.0, -700.0, 120.0]),
            np.asarray([0.0, -700.0, 20.0]),
            rpy_candidates,
            args,
        )

        self.assertTrue(plans)
        self.assertTrue(all(plan.pickup_rotation_deg <= 90.0 for plan in plans))
        self.assertTrue(all(plan.pickup_rotation_deg < 10.0 for plan in plans))
        self.assertEqual({plan.cup.name for plan in plans}, {"primary", "secondary"})
        exact_plan_count = next(
            index
            for index, plan in enumerate(plans)
            if "_singularity_recovery_" in plan.rpy_mode
        )
        self.assertEqual(exact_plan_count, 2)
        self.assertTrue(
            all("_singularity_recovery_" not in plan.rpy_mode for plan in plans[:2])
        )
        self.assertTrue(
            all("_singularity_recovery_" in plan.rpy_mode for plan in plans[2:])
        )
        self.assertGreaterEqual(len(plans[: args.max_controller_plan_attempts]), 3)

    def test_b_origin_still_measures_pickup_rotation_from_a_star(self) -> None:
        candidate = self._candidate("soft_parcel")
        candidate.point_base_mm = np.asarray([0.0, -700.0, 20.0])
        args = build_parser().parse_args([])
        args.force_suction_cups = ["1"]
        args.unused_cup_min_clearance_mm = 0.0
        args.suction_cup_collision_radius_mm = 0.0
        pickup_rpy_deg = np.asarray([0.0, 0.0, -6.0])

        plans = grasp.build_suction_approach_plans(
            candidate,
            [candidate],
            np.asarray([grasp.WAYPOINT_B.x_mm, grasp.WAYPOINT_B.y_mm, grasp.WAYPOINT_B.z_mm]),
            np.asarray([grasp.WAYPOINT_B.rx_deg, grasp.WAYPOINT_B.ry_deg, grasp.WAYPOINT_B.rz_deg]),
            np.asarray([0.0, -700.0, 120.0]),
            np.asarray([0.0, -700.0, 20.0]),
            [("test", pickup_rpy_deg)],
            args,
        )

        self.assertEqual(len(plans), 1)
        self.assertGreater(plans[0].pickup_rotation_deg, 80.0)
        self.assertLess(plans[0].pickup_rotation_deg, 90.0)

    def test_box_plan_with_large_d_turn_uses_rotation_safe_target(self) -> None:
        candidate = self._candidate("parcel_box")
        candidate.point_base_mm = np.asarray([0.0, -700.0, 20.0])
        candidate.short_axis_base = np.asarray([0.0, 1.0, 0.0])
        args = build_parser().parse_args([])
        args.force_suction_cups = ["1"]
        args.unused_cup_min_clearance_mm = 0.0
        args.suction_cup_collision_radius_mm = 0.0

        plans = grasp.build_suction_approach_plans(
            candidate,
            [candidate],
            np.asarray([grasp.WAYPOINT_A_STAR.x_mm, grasp.WAYPOINT_A_STAR.y_mm, grasp.WAYPOINT_A_STAR.z_mm]),
            np.asarray([grasp.WAYPOINT_A_STAR.rx_deg, grasp.WAYPOINT_A_STAR.ry_deg, grasp.WAYPOINT_A_STAR.rz_deg]),
            np.asarray([0.0, -700.0, 120.0]),
            np.asarray([0.0, -700.0, 20.0]),
            [("fixed", np.asarray([0.0, 0.0, -6.0]))],
            args,
        )

        self.assertEqual(len(plans), 1)
        self.assertGreater(abs(plans[0].placement_rotation_deg), 5.0)
        self.assertAlmostEqual(
            plans[0].planned_placement_angle_deg,
            plans[0].placement_rotation_deg,
        )
        self.assertEqual(plans[0].placement_quality, "aligned")
        self.assertAlmostEqual(plans[0].placement_alignment_error_deg, 0.0)

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

    def test_secondary_rotation_is_one_compensated_move(self) -> None:
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

        self.assertEqual(len(commanded_physical_waypoints), 1)
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

        self.assertEqual(len(commanded_offsets), 1)
        self.assertAlmostEqual(commanded_offsets[-1], 0.0, places=6)

    def test_failed_direct_rotation_restores_start_without_boundary_search(self) -> None:
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
            if len(commanded_offsets) == 1 and not failed_once:
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

        np.testing.assert_allclose(commanded_offsets, [80, 0], atol=1e-6)
        self.assertEqual(len(commanded_offsets), 2)
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
        self.assertAlmostEqual(reached_delta, 0.0, places=6)

    def test_near_square_top_uses_whichever_edge_needs_less_d_rotation(self) -> None:
        candidate = self._candidate("parcel_box")
        candidate.short_axis_base = np.asarray([0.0, 1.0, 0.0])
        candidate.top_aspect_ratio = 1.05
        candidate.near_square_top = True

        _safe, _d, square_delta_deg, square_error_deg = grasp.aligned_placement_waypoints(
            candidate,
            np.asarray([0.0, 0.0, 0.0]),
        )
        candidate.near_square_top = False
        _safe, _d, rectangular_delta_deg, _error = grasp.aligned_placement_waypoints(
            candidate,
            np.asarray([0.0, 0.0, 0.0]),
        )

        self.assertLess(abs(square_delta_deg), 1.0)
        self.assertAlmostEqual(square_error_deg, 0.0, places=6)
        self.assertGreater(abs(rectangular_delta_deg), 80.0)

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
