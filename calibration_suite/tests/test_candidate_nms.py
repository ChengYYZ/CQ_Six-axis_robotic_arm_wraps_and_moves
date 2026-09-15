from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

from surface_cluster_grasp import (
    ClusterCandidate,
    candidate_footprint_clearance_mm,
    prune_duplicate_candidates,
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

    def test_near_identical_different_classes_are_preserved(self) -> None:
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

        self.assertEqual({item.index for item in kept}, {1, 2})


if __name__ == "__main__":
    unittest.main()
