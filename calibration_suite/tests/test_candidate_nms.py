from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

from surface_cluster_grasp import (
    ClusterCandidate,
    candidate_footprint_clearance_mm,
    candidate_volume_clearance_mm,
    prune_duplicate_candidates,
    rank_candidate_plan_pairs,
    rank_candidates_for_next_pick,
)


def make_candidate(
    index: int,
    hull: list[list[int]],
    center: tuple[int, int],
    *,
    confidence: float = 0.8,
    class_id: int = 0,
    height_mm: float = 300.0,
    flatness_mm: float = 1.0,
    foreground_ratio: float = 0.9,
) -> ClusterCandidate:
    return ClusterCandidate(
        index=index,
        center_pixel=center,
        hull_pixels=np.asarray(hull, dtype=np.int32),
        support_region_id="yolo",
        support_region_name="YOLO",
        point_camera_mm=np.asarray([0.0, 0.0, 500.0]),
        point_base_mm=np.asarray([0.0, 0.0, height_mm]),
        normal_camera=np.asarray([0.0, 0.0, 1.0]),
        normal_base=np.asarray([0.0, 0.0, 1.0]),
        height_mm=height_mm,
        flatness_mm=flatness_mm,
        point_count=1000,
        rect_area_px=1000.0,
        foreground_ratio=foreground_ratio,
        class_id=class_id,
        class_name="parcel_box",
        confidence=confidence,
    )


class CandidateNmsTests(unittest.TestCase):
    def test_oriented_footprint_clearance_detects_unused_cup_collision(self) -> None:
        candidate = make_candidate(
            1,
            [[10, 10], [190, 10], [190, 190], [10, 190]],
            (100, 100),
        )
        candidate.point_camera_mm = np.asarray([0.0, 0.0, 500.0])
        candidate.point_base_mm = np.asarray([0.0, 0.0, 300.0])
        candidate.short_axis_camera = np.asarray([1.0, 0.0, 0.0])
        candidate.short_axis_base = np.asarray([1.0, 0.0, 0.0])
        candidate.point_cloud_camera_mm = np.asarray(
            [
                [x, y, 500.0]
                for x in (-50.0, 50.0)
                for y in (-200.0, 200.0)
            ]
        )

        clearance = candidate_footprint_clearance_mm(
            candidate,
            np.asarray([80.0, 0.0, 300.0]),
            cup_collision_radius_mm=35.0,
        )

        self.assertLess(clearance, 0.0)

    def test_unused_cup_volume_uses_vertical_clearance_above_target(self) -> None:
        candidate = make_candidate(
            1,
            [[10, 10], [190, 10], [190, 190], [10, 190]],
            (100, 100),
            height_mm=300.0,
        )
        candidate.short_axis_camera = np.asarray([1.0, 0.0, 0.0])
        candidate.short_axis_base = np.asarray([1.0, 0.0, 0.0])
        candidate.point_camera_mm = np.asarray([0.0, 0.0, 500.0])
        candidate.point_base_mm = np.asarray([0.0, 0.0, 300.0])
        candidate.point_cloud_camera_mm = np.asarray(
            [[x, y, 500.0] for x in (-100.0, 100.0) for y in (-100.0, 100.0)]
        )

        near_clearance = candidate_volume_clearance_mm(
            candidate, np.asarray([0.0, 0.0, 340.0]), 35.0
        )
        above_clearance = candidate_volume_clearance_mm(
            candidate, np.asarray([0.0, 0.0, 400.0]), 35.0
        )

        self.assertLess(near_clearance, 20.0)
        self.assertGreaterEqual(above_clearance, 20.0)

    def test_nested_small_thin_package_is_not_suppressed(self) -> None:
        large = make_candidate(
            1,
            [[10, 10], [190, 10], [190, 190], [10, 190]],
            (100, 100),
            confidence=0.95,
            height_mm=300.0,
        )
        small = make_candidate(
            2,
            [[70, 75], [130, 75], [130, 125], [70, 125]],
            (100, 100),
            confidence=0.80,
            height_mm=306.0,
        )

        kept = prune_duplicate_candidates([large, small], (200, 200), 0.75, 0.70, 0.20)

        self.assertEqual({item.index for item in kept}, {1, 2})

    def test_nested_small_thin_package_is_ranked_before_larger_package(self) -> None:
        large = make_candidate(
            1,
            [[10, 10], [190, 10], [190, 190], [10, 190]],
            (100, 100),
            confidence=0.95,
            height_mm=300.0,
        )
        small = make_candidate(
            2,
            [[70, 75], [130, 75], [130, 125], [70, 125]],
            (100, 100),
            confidence=0.80,
            height_mm=306.0,
        )

        ranked, _coverage = rank_candidates_for_next_pick([large, small], (200, 200))

        self.assertEqual([item.index for item in ranked], [2, 1])

    def test_weighted_score_can_outweigh_height_order(self) -> None:
        higher_low_confidence = make_candidate(
            1,
            [[10, 10], [70, 10], [70, 70], [10, 70]],
            (40, 40),
            confidence=0.10,
            height_mm=320.0,
        )
        lower_high_confidence = make_candidate(
            2,
            [[110, 110], [170, 110], [170, 170], [110, 170]],
            (140, 140),
            confidence=0.99,
            height_mm=300.0,
        )

        ranked, _coverage = rank_candidates_for_next_pick(
            [higher_low_confidence, lower_high_confidence],
            (200, 200),
            height_weight=0.10,
            occlusion_weight=0.0,
            flatness_weight=0.0,
            confidence_weight=0.90,
            point_quality_weight=0.0,
        )

        self.assertEqual([item.index for item in ranked], [2, 1])
        self.assertGreater(lower_high_confidence.selection_score, higher_low_confidence.selection_score)

    def test_front_roi_has_hard_priority_over_normal_pick_score(self) -> None:
        front_low_score = make_candidate(
            1,
            [[10, 120], [70, 120], [70, 180], [10, 180]],
            (40, 150),
            confidence=0.10,
            height_mm=280.0,
        )
        rear_high_score = make_candidate(
            2,
            [[110, 10], [170, 10], [170, 70], [110, 70]],
            (140, 40),
            confidence=0.99,
            height_mm=340.0,
        )
        front_roi = np.asarray(
            [[0, 100], [90, 100], [90, 199], [0, 199]], dtype=np.int32
        )

        ranked, _coverage = rank_candidates_for_next_pick(
            [rear_high_score, front_low_score],
            (200, 200),
            front_priority_polygon=front_roi,
        )

        self.assertEqual([item.index for item in ranked], [1, 2])
        self.assertTrue(front_low_score.front_priority)
        self.assertFalse(rear_high_score.front_priority)
        self.assertIn("front_priority=1", front_low_score.selection_note)

    def test_unsafe_front_candidate_does_not_outrank_safe_rear_candidate(self) -> None:
        unsafe_front = make_candidate(
            1,
            [[10, 120], [70, 120], [70, 180], [10, 180]],
            (40, 150),
        )
        unsafe_front.motion_safe = False
        safe_rear = make_candidate(
            2,
            [[110, 10], [170, 10], [170, 70], [110, 70]],
            (140, 40),
        )
        front_roi = np.asarray(
            [[0, 100], [90, 100], [90, 199], [0, 199]], dtype=np.int32
        )

        ranked, _coverage = rank_candidates_for_next_pick(
            [unsafe_front, safe_rear],
            (200, 200),
            front_priority_polygon=front_roi,
        )

        self.assertEqual(ranked[0].index, 2)
        self.assertTrue(unsafe_front.front_priority)

    def test_upper_partial_front_parcel_outranks_covered_full_front_parcel(self) -> None:
        lower_full_front = make_candidate(
            1,
            [[10, 100], [100, 100], [100, 190], [10, 190]],
            (55, 145),
            height_mm=280.0,
        )
        upper_partial_front = make_candidate(
            2,
            [[80, 70], [180, 70], [180, 150], [80, 150]],
            (130, 110),
            height_mm=320.0,
        )
        front_roi = np.asarray(
            [[0, 90], [105, 90], [105, 199], [0, 199]], dtype=np.int32
        )

        ranked, coverage = rank_candidates_for_next_pick(
            [lower_full_front, upper_partial_front],
            (200, 200),
            front_priority_polygon=front_roi,
            max_occlusion_ratio=0.40,
        )

        self.assertTrue(lower_full_front.front_priority)
        self.assertFalse(upper_partial_front.front_priority)
        self.assertTrue(coverage[id(lower_full_front)][0])
        self.assertTrue(lower_full_front.motion_safe)
        self.assertEqual([item.index for item in ranked], [2, 1])

    def test_severe_occlusion_blocks_candidate_from_motion(self) -> None:
        lower = make_candidate(
            1,
            [[10, 10], [110, 10], [110, 110], [10, 110]],
            (60, 60),
            height_mm=280.0,
        )
        upper = make_candidate(
            2,
            [[10, 10], [60, 10], [60, 110], [10, 110]],
            (35, 60),
            height_mm=310.0,
        )

        ranked, coverage = rank_candidates_for_next_pick(
            [lower, upper],
            (140, 140),
            max_occlusion_ratio=0.40,
        )

        self.assertTrue(coverage[id(lower)][0])
        self.assertFalse(lower.motion_safe)
        self.assertIn("occluded=", lower.filter_note)
        self.assertEqual(ranked[0].index, 2)

    def test_near_identical_same_class_detection_is_suppressed(self) -> None:
        better = make_candidate(
            1,
            [[30, 30], [130, 30], [130, 130], [30, 130]],
            (80, 80),
            confidence=0.95,
        )
        duplicate = make_candidate(
            2,
            [[32, 32], [132, 32], [132, 132], [32, 132]],
            (82, 82),
            confidence=0.70,
        )

        kept = prune_duplicate_candidates([duplicate, better], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual([item.index for item in kept], [1])

    def test_near_identical_different_classes_are_merged_by_3d_position(self) -> None:
        first = make_candidate(
            1,
            [[30, 30], [130, 30], [130, 130], [30, 130]],
            (80, 80),
            class_id=0,
        )
        second = make_candidate(
            2,
            [[31, 31], [131, 31], [131, 131], [31, 131]],
            (81, 81),
            class_id=1,
        )

        kept = prune_duplicate_candidates([first, second], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual(len(kept), 1)

    def test_near_center_different_size_and_class_is_one_surface(self) -> None:
        larger = make_candidate(
            1, [[20, 20], [140, 20], [140, 140], [20, 140]],
            (80, 80), confidence=0.76, class_id=0, height_mm=106.1,
        )
        smaller = make_candidate(
            2, [[40, 40], [120, 40], [120, 120], [40, 120]],
            (82, 79), confidence=0.63, class_id=2, height_mm=105.8,
        )
        smaller.point_base_mm = np.asarray([2.0, 4.0, 105.8])

        kept = prune_duplicate_candidates([smaller, larger], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual([item.index for item in kept], [1])

    def test_near_center_same_class_at_different_heights_is_preserved(self) -> None:
        lower = make_candidate(
            1, [[20, 20], [140, 20], [140, 140], [20, 140]],
            (80, 80), height_mm=100.0,
        )
        upper = make_candidate(
            2, [[22, 22], [138, 22], [138, 138], [22, 138]],
            (80, 80), height_mm=140.0,
        )

        kept = prune_duplicate_candidates([lower, upper], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual({item.index for item in kept}, {1, 2})

    def test_near_center_rotated_rectangles_on_one_surface_are_suppressed(self) -> None:
        horizontal = make_candidate(
            1, [[20, 40], [140, 40], [140, 120], [20, 120]],
            (80, 80), confidence=0.9,
        )
        vertical = make_candidate(
            2, [[40, 20], [120, 20], [120, 140], [40, 140]],
            (80, 80), confidence=0.7,
        )

        kept = prune_duplicate_candidates([vertical, horizontal], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual([item.index for item in kept], [1])

    def test_overlapping_different_classes_at_different_heights_are_preserved(self) -> None:
        lower = make_candidate(
            1,
            [[30, 30], [130, 30], [130, 130], [30, 130]],
            (80, 80),
            class_id=0,
            height_mm=280.0,
        )
        upper = make_candidate(
            2,
            [[31, 31], [131, 31], [131, 131], [31, 131]],
            (81, 81),
            class_id=1,
            height_mm=325.0,
        )

        kept = prune_duplicate_candidates([lower, upper], (180, 180), 0.75, 0.70, 0.20)

        self.assertEqual({item.index for item in kept}, {1, 2})

    def test_joint_ranking_skips_candidate_without_a_feasible_plan(self) -> None:
        front_without_plan = make_candidate(
            1,
            [[10, 120], [70, 120], [70, 180], [10, 180]],
            (40, 150),
        )
        front_without_plan.front_priority = True
        front_without_plan.selection_score = 0.95
        rear_with_plan = make_candidate(
            2,
            [[110, 10], [170, 10], [170, 70], [110, 70]],
            (140, 40),
        )
        rear_with_plan.selection_score = 0.70
        plan = SimpleNamespace(
            score=500.0,
            placement_quality="aligned",
            placement_alignment_error_deg=0.0,
            cup=SimpleNamespace(name="primary"),
        )

        ranked = rank_candidate_plan_pairs(
            [front_without_plan, rear_with_plan],
            {front_without_plan.index: [], rear_with_plan.index: [plan]},
        )

        self.assertEqual([candidate.index for candidate in ranked], [2, 1])

    def test_trajectory_cost_does_not_override_grasp_ranking(self) -> None:
        better_grasp = make_candidate(
            1,
            [[10, 120], [70, 120], [70, 180], [10, 180]],
            (40, 150),
        )
        better_grasp.selection_score = 0.90
        worse_grasp = make_candidate(
            2,
            [[110, 120], [170, 120], [170, 180], [110, 180]],
            (140, 150),
        )
        worse_grasp.selection_score = 0.50
        expensive_plan = SimpleNamespace(
            score=2000.0,
            placement_quality="degraded",
            placement_alignment_error_deg=20.0,
            cup=SimpleNamespace(name="primary"),
        )
        cheap_plan = SimpleNamespace(
            score=200.0,
            placement_quality="aligned",
            placement_alignment_error_deg=0.0,
            cup=SimpleNamespace(name="secondary"),
        )

        ranked = rank_candidate_plan_pairs(
            [better_grasp, worse_grasp],
            {better_grasp.index: [expensive_plan], worse_grasp.index: [cheap_plan]},
        )

        self.assertEqual([candidate.index for candidate in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
