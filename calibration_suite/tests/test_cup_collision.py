import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project0714_grasp.cup_collision import check_cup_volume
from project0714_calib.common import rpy_xyz_to_matrix
import surface_cluster_grasp as grasp


class CupCollisionTests(unittest.TestCase):
    def test_dimensions_contact_and_body_side(self):
        points = np.array([[0, 0, 0], [0, 0, 2], [39.8, 29.8, 43],
                           [39.9, 0, 10], [0, 29.9, 10], [0, 0, 43.1],
                           [0, 0, -10], [np.nan, 0, 5]])
        result = check_cup_volume(points, np.zeros(3), np.eye(3))
        self.assertEqual(result.hit_indices.tolist(), [2])
        self.assertEqual(result.valid_points, 7)
        self.assertEqual(result.max_depth_mm, 43)
        reverse = check_cup_volume(points, np.zeros(3), np.eye(3), body_z_sign=-1)
        self.assertEqual(reverse.hit_indices.tolist(), [6])

    def test_rotated_translated_box(self):
        rotation = rpy_xyz_to_matrix(np.radians([31, -54, 127]))
        center = np.array([610, 240, 430])
        local = np.array([[30, 20, 15], [0, 31, 20]])
        result = check_cup_volume(local @ rotation.T + center, center, rotation)
        self.assertEqual(result.hit_indices.tolist(), [0])

    def test_empty_is_unknown_and_bad_rotation_rejected(self):
        result = check_cup_volume(np.empty((0, 3)), np.zeros(3), np.eye(3))
        self.assertEqual(result.valid_points, 0)
        with self.assertRaises(ValueError):
            check_cup_volume([[0, 0, 3]], np.zeros(3), np.zeros((3, 3)))

    def test_unused_offset_perpendicular_cup_is_checked(self):
        args = SimpleNamespace(cup_volume_check=True, tool_contact_axis="minus-z",
                               tool_face_reference_rpy_deg=None, cup_volume_min_points=1)
        cup = grasp.SuctionCupSpec("third", 4, np.array([100., 0, 113]),
                                  rpy_xyz_to_matrix(np.radians([0, -90, 0])))
        candidate = SimpleNamespace(collision_scene_base_mm=np.array([[90., 0, 113]]),
                                    collision_target_surface_base_mm=np.array([[999., 999., 999.]]))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            self.assertFalse(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args))
            candidate.collision_scene_base_mm = np.array([[150., 0, 113]])
            self.assertTrue(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args))
            candidate.collision_scene_base_mm = None
            self.assertFalse(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args))

    def test_scene_transform_mm_and_shared_unmasked_cloud(self):
        args = SimpleNamespace(cup_volume_check=True, min_depth_mm=1, max_depth_mm=1000)
        candidates = [SimpleNamespace(), SimpleNamespace()]
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        grasp.attach_collision_scene(candidates, np.array([[100., 0]]), np.eye(3), transform, args)
        np.testing.assert_allclose(candidates[0].collision_scene_base_mm, [[1000, 2000, 3100]])
        self.assertIs(candidates[0].collision_scene_base_mm, candidates[1].collision_scene_base_mm)

    def test_selected_cup_excludes_complete_target_instance_mask(self):
        args = SimpleNamespace(cup_volume_check=True, min_depth_mm=1, max_depth_mm=1000,
                               tool_contact_axis="minus-z", tool_face_reference_rpy_deg=None,
                               standoff_mm=100, pickup_down_mm=110,
                               cup_volume_min_points=1, cup_contact_surface_match_mm=3)
        target_mask = np.array([[True, False, False, False]], dtype=bool)
        candidate = SimpleNamespace(
            collision_target_mask=target_mask,
            point_cloud_camera_mm=np.array([[0., 0., 15.]]),
        )
        grasp.attach_collision_scene(
            [candidate], np.array([[15., 0., 0., 15.]]), np.eye(3), np.eye(4), args
        )
        cup = grasp.SuctionCupSpec("primary", 5, np.zeros(3))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            # The selected cup may overlap its own parcel at the contact
            # endpoint even when that parcel is not a RANSAC top-plane inlier.
            self.assertTrue(
                grasp.pickup_cup_volume_clear(
                    candidate, np.zeros(3), np.eye(3), args, "primary"
                )
            )
            # The same target point remains an obstacle for an unused cup.
            self.assertFalse(
                grasp.pickup_cup_volume_clear(
                    candidate, np.zeros(3), np.eye(3), args, "secondary"
                )
            )

    def test_target_mask_does_not_exclude_external_obstacle(self):
        args = SimpleNamespace(cup_volume_check=True, min_depth_mm=1, max_depth_mm=1000,
                               tool_contact_axis="minus-z", tool_face_reference_rpy_deg=None,
                               standoff_mm=100, pickup_down_mm=110,
                               cup_volume_min_points=1, cup_contact_surface_match_mm=3)
        candidate = SimpleNamespace(
            collision_target_mask=np.array([[True, False]], dtype=bool),
            point_cloud_camera_mm=np.array([[0., 0., 15.]]),
        )
        grasp.attach_collision_scene(
            [candidate], np.array([[15., 15.]]), np.eye(3), np.eye(4), args
        )
        cup = grasp.SuctionCupSpec("primary", 5, np.zeros(3))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            self.assertFalse(
                grasp.pickup_cup_volume_clear(
                    candidate, np.zeros(3), np.eye(3), args, "primary"
                )
            )

    def test_planned_compression_only_exempts_selected_cup(self):
        args = SimpleNamespace(cup_volume_check=True, tool_contact_axis="minus-z",
                               tool_face_reference_rpy_deg=None, standoff_mm=100, pickup_down_mm=105,
                               cup_volume_min_points=1, cup_contact_surface_match_mm=3)
        cup = grasp.SuctionCupSpec("primary", 5, np.zeros(3))
        candidate = SimpleNamespace(collision_scene_base_mm=np.array([[0., 0, 5]]),
                                    collision_target_surface_base_mm=np.array([[99., 99., 99.]]),
                                    collision_target_surface_tree=grasp.cKDTree([[99., 99., 99.]]))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            self.assertTrue(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "primary"))
            self.assertFalse(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "secondary"))
            candidate.collision_scene_base_mm = np.array([[0., 0, 8]])
            self.assertFalse(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "primary"))

    def test_selected_surface_excluded_but_unused_cup_still_collides(self):
        args = SimpleNamespace(cup_volume_check=True, tool_contact_axis="minus-z",
                               tool_face_reference_rpy_deg=None, standoff_mm=100, pickup_down_mm=100,
                               cup_volume_min_points=2, cup_contact_surface_match_mm=3)
        cup = grasp.SuctionCupSpec("primary", 5, np.zeros(3))
        cloud = np.array([[0., 0, 10.], [.5, .5, 10.]])
        candidate = SimpleNamespace(collision_scene_base_mm=cloud,
                                    collision_target_surface_base_mm=cloud.copy(),
                                    collision_target_surface_tree=grasp.cKDTree(cloud.copy()))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            self.assertTrue(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "primary"))
            self.assertFalse(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "secondary"))

    def test_sparse_noise_below_threshold_does_not_reject(self):
        args = SimpleNamespace(cup_volume_check=True, tool_contact_axis="minus-z",
                               tool_face_reference_rpy_deg=None, standoff_mm=100, pickup_down_mm=100,
                               cup_volume_min_points=5, cup_contact_surface_match_mm=3)
        cup = grasp.SuctionCupSpec("primary", 5, np.zeros(3))
        candidate = SimpleNamespace(collision_scene_base_mm=np.array([[0., 0, 10.]]),
                                    collision_target_surface_base_mm=np.array([[99., 99., 99.]]),
                                    collision_target_surface_tree=grasp.cKDTree([[99., 99., 99.]]))
        with patch.object(grasp, "suction_cup_specs", return_value=[cup]):
            self.assertTrue(grasp.pickup_cup_volume_clear(candidate, np.zeros(3), np.eye(3), args, "secondary"))

    def test_ranked_plan_wrapper_checks_exact_selected_pose(self):
        cup = grasp.SuctionCupSpec("secondary", 5, np.zeros(3))
        plan = grasp.SuctionApproachPlan(
            cup=cup,
            approach_xyz_mm=np.array([1., 2., 3.]),
            pickup_xyz_mm=np.array([4., 5., 6.]),
            rpy_deg=np.array([10., 20., 30.]),
            rpy_mode="test",
            score=1., travel_mm=2., cup_to_package_mm=3., radial_reach_mm=4.,
            unused_cup_clearance_mm=5.,
        )
        with patch.object(grasp, "pickup_cup_volume_clear", return_value=True) as check:
            self.assertTrue(grasp.suction_plan_volume_clear(SimpleNamespace(), plan, SimpleNamespace()))
        self.assertEqual(check.call_args.args[4], "secondary")
        np.testing.assert_array_equal(check.call_args.args[1], [4., 5., 6.])
        np.testing.assert_allclose(
            check.call_args.args[2], rpy_xyz_to_matrix(np.radians([10., 20., 30.])), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
