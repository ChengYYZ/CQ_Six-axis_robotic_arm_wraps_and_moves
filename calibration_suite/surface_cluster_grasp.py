from __future__ import annotations

import argparse
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from project0714_calib.common import load_json, matrix_to_rpy_xyz, rpy_xyz_to_matrix, save_json
from project0714_calib.orbbec_camera import CameraOpenOptions, OrbbecRGBDCamera
from project0714_calib.xcore_robot import MotionOptions, XCoreRobotClient
from project0714_grasp.foreground import (
    ForegroundModel as GraspForegroundModel,
    build_foreground_from_support_planes as build_grasp_foreground_from_support_planes,
    compute_height_gradient as compute_foreground_height_gradient,
)
from project0714_grasp.support_planes import (
    PlaneModel as GraspPlaneModel,
    RoiConfig as GraspRoiConfig,
    SupportPlaneModel as GraspSupportPlaneModel,
    SupportRegionSpec as GraspSupportRegionSpec,
    SUPPORT_REGION_SPECS as GRASP_SUPPORT_REGION_SPECS,
    build_support_plane_model as build_grasp_support_plane_model,
    delete_roi_config as delete_grasp_roi_config,
    get_support_region_spec as get_grasp_support_region_spec,
    load_roi_config as load_grasp_roi_config,
    polygon_from_mask as polygon_from_grasp_mask,
    roi_mask_from_polygon as roi_mask_from_grasp_polygon,
    save_roi_config as save_grasp_roi_config,
)
from project0714_grasp.waybill_inspection import AsyncWaybillInspector, WaybillInspectionResult
from project0714_grasp.cup_collision import check_cup_volume


COLOR_WINDOW = "Project0714 Surface Grasp Color"
DEPTH_WINDOW = "Project0714 Surface Grasp Depth"
POINT_CLOUD_WINDOW = "Project0714 Candidate Point Cloud"


class GlobalSpaceStopMonitor:
    """Watch the Windows SPACE key independently of OpenCV window focus."""

    VK_SPACE = 0x20

    def __init__(self, on_space_pressed: Callable[[], None], poll_interval_s: float = 0.02):
        self._on_space_pressed = on_space_pressed
        self._poll_interval_s = poll_interval_s
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._get_async_key_state = None

    def start(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes

            self._get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        except Exception as exc:
            print(f"Warning: global SPACE stop listener is unavailable: {exc}")
            return False

        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="robot-space-stop-monitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        was_pressed = False
        while not self._shutdown.wait(self._poll_interval_s):
            get_key_state = self._get_async_key_state
            if get_key_state is None:
                return
            pressed = bool(get_key_state(self.VK_SPACE) & 0x8000)
            if pressed and not was_pressed:
                try:
                    self._on_space_pressed()
                except Exception as exc:
                    print(f"Warning: SPACE stop callback failed: {exc}")
            was_pressed = pressed

@dataclass
class PlaneModel:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_indices: np.ndarray


@dataclass
class ClusterCandidate:
    index: int
    center_pixel: tuple[int, int]
    hull_pixels: np.ndarray
    support_region_id: str
    support_region_name: str
    point_camera_mm: np.ndarray
    point_base_mm: np.ndarray
    normal_camera: np.ndarray
    normal_base: np.ndarray
    height_mm: float
    flatness_mm: float
    point_count: int
    rect_area_px: float = 0.0
    foreground_ratio: float = 0.0
    detection_source: str = "unknown"
    class_id: int | None = None
    class_name: str = "unknown"
    confidence: float = 0.0
    motion_safe: bool = True
    filter_note: str = "ok"
    point_cloud_camera_mm: np.ndarray | None = None
    short_axis_camera: np.ndarray | None = None
    short_axis_base: np.ndarray | None = None
    selection_score: float = 0.0
    selection_note: str = "not_scored"
    # Shared unsegmented visible scene, in base mm, from this detection frame.
    collision_scene_base_mm: np.ndarray | None = field(default=None, repr=False)
    # Per-pixel detector ownership for this package. The selected suction cup
    # is expected to overlap these points at contact, while every other cup
    # must continue to treat them as obstacles.
    collision_target_mask: np.ndarray | None = field(default=None, repr=False)
    collision_obstacle_scene_base_mm: np.ndarray | None = field(default=None, repr=False)
    collision_target_surface_base_mm: np.ndarray | None = field(default=None, repr=False)
    collision_target_surface_tree: object | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RobotWaypoint:
    name: str
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float


@dataclass
class ApproachPlan:
    candidate_index: int
    approach_xyz_mm: np.ndarray
    pickup_xyz_mm: np.ndarray
    rpy_deg: np.ndarray
    rpy_mode: str
    suction_name: str = "primary"
    suction_do_port: int = 5


@dataclass(frozen=True)
class SuctionCupSpec:
    name: str
    do_port: int
    offset_tool_mm: np.ndarray
    # Rotation from the cup's virtual frame into the active/main TCP frame.
    rotation_tool_from_cup: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )


@dataclass
class SuctionApproachPlan:
    cup: SuctionCupSpec
    approach_xyz_mm: np.ndarray
    pickup_xyz_mm: np.ndarray
    rpy_deg: np.ndarray
    rpy_mode: str
    score: float
    travel_mm: float
    cup_to_package_mm: float
    radial_reach_mm: float
    unused_cup_clearance_mm: float
    placement_rotation_deg: float = 0.0
    placement_tcp_reach_mm: float = 0.0
    rotation_safe_tcp_reach_mm: float = 0.0


class SafeRotationRecoveredError(RuntimeError):
    """A loaded optional rotation failed but was restored to its start angle."""


class BatchExecutionResult(Enum):
    COMPLETED = "completed"
    NO_SAFE_PLAN = "no_safe_plan"
    MOTION_ERROR = "motion_error"


# Fixed safe path waypoints, expressed in the active wobj0/tool coordinate setup.
# A* is the newly taught lift-and-clear transition point used on the loaded return path.
WAYPOINT_A_STAR = RobotWaypoint("A*", 110.628, -829.682, 482.697, -1.917, 2.748, -94.661)
# Physically taught loaded transition for the perpendicular third/fourth cups.
# After retracting to dynamic A, loaded side cups must reach this complete pose
# before moving to B.
WAYPOINT_SIDE_A_STAR = RobotWaypoint(
    "side-loaded-transition",
    -146.465,
    -1076.584,
    656.248,
    152.793,
    87.413,
    -115.695,
)
# Physically required loaded orientation for the perpendicular third/fourth
# cups at the taught transition and through B.
# With rotation_tool_from_cup=Ry(-90), the cup contact direction is the main
# TCP +X axis; at this pose it is approximately [0, 0, -1] in base coordinates.
SIDE_CUP_FACE_DOWN_RPY_DEG = np.asarray(
    [152.793, 87.413, -115.695],
    dtype=np.float64,
)
# B is a taught active-TCP clearance pose.
WAYPOINT_B = RobotWaypoint("B", 779.155, -218.594, 487.419, -0.296, 1.428, -6.094)
# Every C-point route uses the active tool4 TCP pose verified by
# test_c_cup_z_rotation.py. Keep one source of truth for loaded inspection,
# side-view return, and empty C pass-through moves.
WAYPOINT_C = RobotWaypoint("C", 627.094, 311.420, 470.689, -2.136, 3.046, -3.744)
WAYPOINT_C_BARCODE = WAYPOINT_C
WAYPOINT_C_BARCODE_SIDE = RobotWaypoint(
    "C-barcode-side", 627.067, 311.427, 470.629, -0.736, 3.143, 75.357
)
# Relative rotation from the C-point pose used in the verified barcode-camera
# test to its taught side-view pose.  This is deliberately not rounded to a
# nominal 90 degrees: the taught observation pose contains small Rx/Ry terms.
C_BARCODE_SIDE_VIEW_ROTATION = np.asarray(
    [
        [0.19144061, -0.97994523, -0.05529781],
        [0.98147042, 0.19066251, 0.01906899],
        [-0.00814335, -0.05792374, 0.99828780],
    ],
    dtype=np.float64,
)
WAYPOINT_D = RobotWaypoint("D", 452.368, 808.428, 284.504, -3.030, 0.839, 79.567)
# This high point has been physically tested with a 90-degree local-Z turn.
# Its taught orientation is the zero-angle reference for the same tool/package
# relationship used at D. Rotation is completed here before descending to D.
WAYPOINT_D_ROTATE_SAFE = RobotWaypoint(
    "D-rotate-safe", 225.733, 550.238, 535.240, -2.080, 1.287, 83.579
)
FIXED_GRASP_X_YAW_DEG = -96.09


@dataclass
class AnalysisResult:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    depth_display: np.ndarray
    base_plane: PlaneModel | None
    candidates: list[ClusterCandidate]
    notes: list[str]
    debug_bgr: np.ndarray | None = None


@dataclass
class SeedRegion:
    label: int
    area_px: int
    center_pixel: np.ndarray
    top_height_mm: float
    bbox_xyxy: np.ndarray
    pixels: np.ndarray
    points: np.ndarray
    heights: np.ndarray


@dataclass
class RgbRectRegion:
    label: int
    source: str
    contour_area_px: float
    rect_area_px: float
    rectangularity: float
    center_pixel: np.ndarray
    hull_pixels: np.ndarray
    mask: np.ndarray
    point_mask: np.ndarray | None = None
    class_id: int | None = None
    class_name: str = "unknown"
    confidence: float = 0.0


@dataclass
class SupportRegionAnalysis:
    region_id: str
    region_name: str
    plane: PlaneModel | None
    candidates: list[ClusterCandidate]
    notes: list[str]


@dataclass(frozen=True)
class SupportRegionSpec:
    region_id: str
    key_hint: str
    name: str
    color_bgr: tuple[int, int, int]


@dataclass
class RoiConfig:
    overall_polygon: np.ndarray | None
    support_polygons: dict[str, np.ndarray]
    exclude_polygons: list[np.ndarray]
    suction_zone_polygons: dict[str, np.ndarray] = field(default_factory=dict)


SUPPORT_REGION_SPECS: tuple[SupportRegionSpec, ...] = (
    SupportRegionSpec("floor", "1", "Floor", (255, 200, 0)),
    SupportRegionSpec("left_wall", "2", "Left Wall", (255, 120, 0)),
    SupportRegionSpec("right_wall", "3", "Right Wall", (0, 220, 255)),
    SupportRegionSpec("back_wall", "4", "Back Wall", (180, 80, 255)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project0714 surface segmentation + plane fitting + click-to-move demo"
    )
    parser.add_argument(
        "--intrinsics",
        default=str(Path(__file__).resolve().parent / "workspace" / "intrinsics" / "camera_intrinsics.json"),
        help="Camera intrinsics JSON path",
    )
    parser.add_argument(
        "--hand-eye",
        default=str(Path(__file__).resolve().parent / "workspace" / "eye_to_hand" / "eye_to_hand_result.json"),
        help="Eye-to-hand JSON path",
    )
    parser.add_argument(
        "--roi-json",
        default=str(Path(__file__).resolve().parent / "workspace" / "surface_grasp" / "manual_roi.json"),
        help="Manual tray ROI polygon JSON path",
    )
    parser.add_argument(
        "--debug-image",
        default=str(Path(__file__).resolve().parent / "workspace" / "surface_grasp" / "debug_masks.png"),
        help="Path where the latest debug mask montage is saved",
    )
    parser.add_argument(
        "--point-cloud-debug-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "surface_grasp" / "point_cloud_debug"),
        help="Directory for clicked candidate point-cloud PNG/PLY debug files",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo", "rgb"),
        default="yolo",
        help="Package detector: YOLO model or legacy RGB/depth heuristics",
    )
    parser.add_argument(
        "--yolo-model",
        default=str(
            Path(__file__).resolve().parents[1]
            / "project_yolo_train_0804"
            / "runs"
            / "yolo11s_obb_2.0"
            / "weights"
            / "best.pt"
        ),
        help="YOLO model path, supports OBB, segmentation, or boxes",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--yolo-point-erode-px", type=int, default=3, help="Erode YOLO region before point-cloud fitting")
    parser.add_argument("--pick-height-weight", type=float, default=0.35, help="Next-pick score weight for normalized base-frame Z")
    parser.add_argument("--pick-occlusion-weight", type=float, default=0.30, help="Next-pick score weight for visible/uncovered area")
    parser.add_argument("--pick-flatness-weight", type=float, default=0.15, help="Next-pick score weight for top-surface flatness")
    parser.add_argument("--pick-confidence-weight", type=float, default=0.10, help="Next-pick score weight for YOLO confidence")
    parser.add_argument("--pick-point-quality-weight", type=float, default=0.10, help="Next-pick score weight for valid depth-pixel ratio")
    parser.add_argument("--pick-nested-small-bonus", type=float, default=0.05, help="Score bonus for a substantially smaller nested parcel")
    parser.add_argument(
        "--pick-max-occlusion-ratio",
        type=float,
        default=0.40,
        help="Reject a candidate when a package at least 10 mm higher covers this fraction of it",
    )
    parser.add_argument(
        "--enable-waybill-inspection",
        action="store_true",
        help="Inspect the carried package bottom with the Hikvision camera while moving through C",
    )
    parser.add_argument("--waybill-camera-ip", default="192.168.2.14", help="Hikvision bottom camera IP")
    parser.add_argument("--waybill-camera-username", default="admin", help="Hikvision camera username")
    parser.add_argument(
        "--waybill-camera-password",
        default=os.environ.get("HIKVISION_PASSWORD", ""),
        help="Hikvision password; prefer the HIKVISION_PASSWORD environment variable",
    )
    parser.add_argument(
        "--waybill-model",
        default=str(Path(__file__).resolve().parents[1] / "weights" / "best.pt"),
        help="YOLO model used to detect a waybill on the package bottom",
    )
    parser.add_argument("--waybill-conf", type=float, default=0.5, help="Waybill YOLO confidence threshold")
    parser.add_argument(
        "--barcode-model",
        default=str(
            Path(__file__).resolve().parents[1]
            / "weights"
            / "barcode_weights"
            / "best.pt"
        ),
        help="YOLO OBB model used to locate Code128 barcodes in the waybill ROI",
    )
    parser.add_argument(
        "--barcode-conf",
        type=float,
        default=0.25,
        help="Barcode OBB confidence threshold",
    )
    parser.add_argument("--waybill-capture-count", type=int, default=15, help="Maximum settled C-point frames retained for inspection")
    parser.add_argument("--waybill-capture-interval-s", type=float, default=0.1, help="Background snapshot interval")
    parser.add_argument("--waybill-start-delay-s", type=float, default=0.5, help="Delay after waypoint B before bottom-camera capture starts")
    parser.add_argument("--waybill-capture-duration-s", type=float, default=30.0, help="Safety timeout from B prewarm through completion of C capture")
    parser.add_argument("--waybill-c-settle-s", type=float, default=0.6, help="Discard frames for this long after reaching C to let motion/exposure settle")
    parser.add_argument("--waybill-post-c-capture-s", type=float, default=2.5, help="Capture settled stationary frames for this long at C")
    parser.add_argument("--waybill-c-dwell-s", type=float, default=3.2, help="Minimum hold at C; automatically extended to cover settle plus capture")
    parser.add_argument(
        "--waybill-result-timeout-s",
        type=float,
        default=20.0,
        help="Maximum time to wait for one C-point recognition result before treating it as unreadable",
    )
    parser.add_argument("--waybill-request-timeout-s", type=float, default=1.0, help="Per-channel snapshot timeout")
    parser.add_argument(
        "--waybill-output-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "waybill_inspection"),
        help="Directory for bottom-camera evidence frames and waybill crops",
    )
    parser.add_argument("--robot-ip", default="192.168.2.160", help="Robot controller IP")
    parser.add_argument(
        "--tool-name",
        default=None,
        help="RobotAssist/controller tool name to use as the active suction TCP, e.g. suction1",
    )
    parser.add_argument(
        "--tcp-override-xyz-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Session-only TCP XYZ override in mm; does not modify the controller tool record",
    )
    parser.add_argument(
        "--tcp-override-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Session-only TCP RPY override in degrees; does not modify the controller tool record",
    )
    parser.add_argument(
        "--wobj-name",
        default="wobj0",
        help="RobotAssist/controller work object name paired with --tool-name",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not command the robot")
    parser.add_argument(
        "--control-stdin",
        action="store_true",
        help="Accept GUI control commands (ANALYZE, START, RESET, STOP, QUIT) on stdin",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open OpenCV display windows")
    parser.add_argument("--gui-frame-dir", default=None, help="Publish live color/depth preview JPEGs for a GUI")
    parser.add_argument("--gui-frame-interval-s", type=float, default=0.20, help="Minimum interval between GUI preview updates")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=800, help="Color stream height")
    parser.add_argument("--fps", type=int, default=10, help="Color stream FPS")
    parser.add_argument("--depth-width", type=int, default=640, help="Depth stream width")
    parser.add_argument("--depth-height", type=int, default=400, help="Depth stream height")
    parser.add_argument("--depth-fps", type=int, default=5, help="Depth stream FPS")
    parser.add_argument("--align-mode", choices=("sw", "hw", "off"), default="sw", help="RGB-D align mode")
    parser.add_argument("--wait-timeout-ms", type=int, default=500, help="Frame wait timeout")
    parser.add_argument("--sample-step", type=int, default=4, help="Pixel sampling step for point cloud")
    parser.add_argument("--min-depth-mm", type=float, default=350.0, help="Minimum valid depth")
    parser.add_argument("--max-depth-mm", type=float, default=1600.0, help="Maximum valid depth")
    parser.add_argument("--plane-threshold-mm", type=float, default=6.0, help="Bottom-plane inlier threshold")
    parser.add_argument("--plane-iterations", type=int, default=250, help="Bottom-plane RANSAC iterations")
    parser.add_argument("--top-plane-ransac-threshold-mm", type=float, default=3.0,
                        help="Package-top RANSAC point-to-plane inlier threshold; default 3 mm")
    parser.add_argument("--top-plane-ransac-iterations", type=int, default=150,
                        help="Package-top RANSAC iterations before SVD refinement; default 150")
    parser.add_argument("--object-height-mm", type=float, default=10.0, help="Minimum height above bottom plane")
    parser.add_argument("--cluster-tolerance-mm", type=float, default=28.0, help="3D Euclidean clustering radius")
    parser.add_argument("--min-cluster-points", type=int, default=35, help="Minimum sampled points per object")
    parser.add_argument(
        "--max-cluster-points",
        type=int,
        default=250000,
        help="Maximum foreground points per package instance",
    )
    parser.add_argument("--top-slice-mm", type=float, default=12.0, help="Thickness used to extract top surface")
    parser.add_argument("--max-flatness-mm", type=float, default=4.5, help="Maximum mean residual for top plane")
    parser.add_argument("--min-top-points", type=int, default=80, help="Minimum top-surface points")
    parser.add_argument("--min-component-area-px", type=int, default=300, help="Minimum image-space area per object")
    parser.add_argument("--min-seed-area-px", type=int, default=60, help="Minimum image-space area per top seed")
    parser.add_argument("--mask-open-px", type=int, default=2, help="Opening kernel size for the object mask")
    parser.add_argument("--mask-close-px", type=int, default=1, help="Closing kernel size for the object mask")
    parser.add_argument("--tray-erode-px", type=int, default=15, help="Pixels eroded from the tray footprint edge")
    parser.add_argument(
        "--max-top-gradient-mm-per-px",
        type=float,
        default=6.0,
        help="Maximum local height gradient allowed for top-surface seeds",
    )
    parser.add_argument(
        "--top-mask-close-px",
        type=int,
        default=2,
        help="Closing kernel size for the top-surface seed mask",
    )
    parser.add_argument("--merge-distance-px", type=float, default=48.0, help="Maximum pixel distance for merging top seeds")
    parser.add_argument("--merge-height-mm", type=float, default=18.0, help="Maximum top-height difference for merging top seeds")
    parser.add_argument("--merge-gap-px", type=float, default=12.0, help="Maximum bbox gap for merging nearby seed regions")
    parser.add_argument("--split-peak-threshold-ratio", type=float, default=0.42, help="Distance-transform peak ratio used to split touching boxes")
    parser.add_argument("--split-peak-min-distance-px", type=float, default=26.0, help="Minimum peak-center spacing required to split one connected seed")
    parser.add_argument("--rgb-canny-low", type=int, default=35, help="Low threshold for RGB edge detection")
    parser.add_argument("--rgb-canny-high", type=int, default=110, help="High threshold for RGB edge detection")
    parser.add_argument("--rgb-min-saturation", type=int, default=24, help="Minimum HSV saturation used to separate cardboard from gray tray")
    parser.add_argument("--rgb-edge-close-px", type=int, default=5, help="Closing radius for RGB rectangle masks")
    parser.add_argument("--rgb-edge-dilate-px", type=int, default=2, help="Dilation radius for RGB rectangle edges")
    parser.add_argument("--rgb-color-close-px", type=int, default=18, help="Closing radius for color-difference package masks")
    parser.add_argument("--rgb-color-depth-close-px", type=int, default=4, help="Closing radius for color-and-depth package masks")
    parser.add_argument("--seed-color-expand-px", type=int, default=60, help="Pixels used to grow a depth seed into nearby package-color pixels")
    parser.add_argument("--seed-color-close-px", type=int, default=5, help="Closing radius when expanding depth seeds on RGB package-color masks")
    parser.add_argument("--rgb-background-max-saturation", type=int, default=42, help="Maximum saturation used to estimate the gray tray background")
    parser.add_argument("--rgb-lab-distance", type=float, default=16.0, help="Minimum LAB color distance from the gray tray background")
    parser.add_argument("--rgb-cardboard-hue-min", type=int, default=3, help="Minimum HSV hue for cardboard-like pixels")
    parser.add_argument("--rgb-cardboard-hue-max", type=int, default=42, help="Maximum HSV hue for cardboard-like pixels")
    parser.add_argument("--rgb-cardboard-strict-saturation", type=int, default=35, help="Minimum HSV saturation for strong cardboard candidates")
    parser.add_argument("--rgb-cardboard-value-min", type=int, default=55, help="Minimum HSV value for strong cardboard candidates")
    parser.add_argument("--rgb-cardboard-value-max", type=int, default=245, help="Maximum HSV value for strong cardboard candidates")
    parser.add_argument("--rgb-min-rect-area-px", type=float, default=2500.0, help="Minimum RGB rotated-rectangle area")
    parser.add_argument("--rgb-max-rect-area-px", type=float, default=65000.0, help="Maximum RGB rotated-rectangle area")
    parser.add_argument("--rgb-min-rect-side-px", type=float, default=28.0, help="Minimum side length of an RGB rectangle")
    parser.add_argument("--rgb-max-rect-side-px", type=float, default=520.0, help="Maximum side length of an RGB rectangle")
    parser.add_argument("--rgb-min-rectangularity", type=float, default=0.18, help="Minimum contour-area / rectangle-area ratio")
    parser.add_argument("--rgb-max-rect-outside-roi-ratio", type=float, default=0.0, help="Maximum rectangle area allowed outside the active ROI")
    parser.add_argument(
        "--rgb-nms-overlap",
        type=float,
        default=0.65,
        help="Deprecated compatibility option; YOLO regions are no longer removed by containment overlap",
    )
    parser.add_argument("--min-rect-foreground-ratio", type=float, default=0.22, help="Minimum foreground ratio inside an RGB rectangle")
    parser.add_argument("--min-rect-foreground-pixels", type=int, default=250, help="Minimum foreground pixels inside an RGB rectangle")
    parser.add_argument("--max-rect-foreground-pixels", type=int, default=45000, help="Maximum foreground pixels inside one package rectangle")
    parser.add_argument("--min-candidate-height-mm", type=float, default=30.0, help="Minimum height above support plane for a valid package candidate")
    parser.add_argument(
        "--final-candidate-nms-overlap",
        type=float,
        default=0.75,
        help="Final polygon IoU threshold; duplicates must also have similar area and center",
    )
    parser.add_argument(
        "--final-candidate-nms-min-area-ratio",
        type=float,
        default=0.70,
        help="Minimum smaller/larger polygon area ratio for final duplicate suppression",
    )
    parser.add_argument(
        "--final-candidate-nms-max-center-distance",
        type=float,
        default=0.20,
        help="Maximum center distance normalized by mean equivalent polygon size for duplicate suppression",
    )
    parser.add_argument(
        "--grasp-center-mode",
        choices=("yolo-center", "top-centroid"),
        default="yolo-center",
        help="Use detector geometry center or fitted top point centroid as grasp point",
    )
    parser.add_argument(
        "--tool-y-mode",
        choices=("obb-short-edge", "keep-current"),
        default="keep-current",
        help="Deprecated compatibility option; grasp X heading is now fixed and tool/OBB Y is ignored",
    )
    parser.add_argument(
        "--tool-y-tolerance-deg",
        type=float,
        default=10.0,
        help="Yaw tolerance around the selected YOLO short-edge direction when searching for a solvable grasp orientation",
    )
    parser.add_argument(
        "--max-grasp-rotation-deg",
        type=float,
        default=90.0,
        help="Skip short-edge grasp orientations that require a larger rotation from the current tool pose",
    )
    parser.add_argument(
        "--tool-y-try-opposite",
        action="store_true",
        help="Also try the 180-degree opposite short-edge direction; may cause wrist flipping, so keep off for normal use",
    )
    parser.add_argument(
        "--normal-mode",
        choices=("auto", "top-plane", "support-plane", "base-z"),
        default="top-plane",
        help="Normal used for grasp orientation and standoff; default follows the fitted package top plane",
    )
    parser.add_argument(
        "--normal-snap-angle-deg",
        type=float,
        default=30.0,
        help="In auto mode, snap near-horizontal fitted normals to base Z within this angle",
    )
    parser.add_argument("--enable-edge-candidates", action="store_true", help="Also use pure RGB edge rectangles as weak fallback candidates")
    parser.add_argument("--enable-color-only-candidates", action="store_true", help="Also use pure color rectangles as weak fallback candidates")
    parser.add_argument("--enable-package-color-candidates", action="store_true", help="Also use global package-color rectangles as weak fallback candidates")
    parser.add_argument("--disable-cardboard-candidates", action="store_true", help="Disable RGB cardboard-color package candidates")
    parser.add_argument("--workspace-x-min-mm", type=float, default=-450.0, help="Base-frame workspace X min")
    parser.add_argument("--workspace-x-max-mm", type=float, default=600.0, help="Base-frame workspace X max")
    parser.add_argument("--workspace-y-min-mm", type=float, default=-1250.0, help="Base-frame workspace Y min")
    parser.add_argument("--workspace-y-max-mm", type=float, default=-500.0, help="Base-frame workspace Y max")
    parser.add_argument("--workspace-z-min-mm", type=float, default=-50.0, help="Base-frame workspace Z min")
    parser.add_argument("--workspace-z-max-mm", type=float, default=500.0, help="Base-frame workspace Z max")
    parser.add_argument(
        "--ignore-workspace-filter",
        action="store_true",
        help="Ignore workspace bounds for dry-run diagnosis only",
    )
    parser.add_argument(
        "--min-upward-normal-z",
        type=float,
        default=0.7,
        help="Minimum base-frame Z component required for the fitted top-surface normal",
    )
    parser.add_argument("--standoff-mm", type=float, default=100.0, help="Distance kept above the package surface")
    parser.add_argument(
        "--standoff-mode",
        choices=("base-z", "normal"),
        default="normal",
        help="Use base Z or fitted surface normal for the standoff direction",
    )
    parser.add_argument(
        "--rpy-mode",
        choices=("vertical-down", "keep-current", "fixed", "align-normal"),
        default="align-normal",
        help="Robot flange orientation mode for moving above the package",
    )
    parser.add_argument(
        "--fixed-pickup-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Exact commanded TCP RPY used by fixed mode; no pose readback or normal alignment",
    )
    parser.add_argument("--cup-volume-check", action="store_true",
                        help="Reject pickup poses with observed points inside 79.6x59.6x43 mm cup bodies; endpoint only")
    parser.add_argument("--cup-volume-min-points", type=int, default=5,
                        help="Minimum observed points required to declare cup-body interference; default 5")
    parser.add_argument("--cup-contact-surface-match-mm", type=float, default=3.0,
                        help="Selected cup only: ignore points matching its target top surface within this distance")
    parser.add_argument(
        "--tool-contact-axis",
        choices=("plus-z", "minus-z"),
        default="plus-z",
        help=(
            "TCP Z direction containing the suction face: plus-z means TCP +Z points "
            "toward the package; minus-z means TCP -Z points toward the package"
        ),
    )
    parser.add_argument(
        "--tool-face-reference-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Known TCP RPY at which the selected suction face is horizontal and points down",
    )
    parser.add_argument("--vertical-rx-deg", type=float, default=180.0, help="RX used by vertical-down RPY mode")
    parser.add_argument("--vertical-ry-deg", type=float, default=0.0, help="RY used by vertical-down RPY mode")
    parser.add_argument(
        "--vertical-rz-deg",
        type=float,
        default=None,
        help="RZ used by vertical-down RPY mode; default keeps current RZ",
    )
    parser.add_argument(
        "--align-normal-rpy",
        action="store_true",
        help="Deprecated alias: use --rpy-mode align-normal",
    )
    parser.add_argument(
        "--use-current-conf-data",
        action="store_true",
        help="Keep current robot confData when solving target pose; usually leave off if targets report no solution",
    )
    parser.add_argument(
        "--pickup-down-mm",
        type=float,
        default=120.0,
        help="Distance moved from the package approach point down to the manual pickup point",
    )
    parser.add_argument(
        "--pickup-dwell-s",
        type=float,
        default=2.0,
        help="Dwell time after enabling suction at the pickup point",
    )
    parser.add_argument(
        "--place-dwell-s",
        type=float,
        default=2.0,
        help="Delay at D-rotate-safe after releasing the package and before starting the TCP barcode listener",
    )
    parser.add_argument(
        "--verified-placement-angles-deg",
        type=float,
        nargs="+",
        default=(0.0,),
        help=(
            "Local suction-axis placement angles with physically verified loaded routes to D; "
            "only these are tried after the requested aligned D pose fails"
        ),
    )
    parser.add_argument(
        "--primary-verified-placement-angles-deg",
        type=float,
        nargs="+",
        default=None,
        help="Primary-cup verified D angles; defaults to --verified-placement-angles-deg",
    )
    parser.add_argument(
        "--secondary-verified-placement-angles-deg",
        type=float,
        nargs="+",
        default=None,
        help="Secondary-cup verified D angles; defaults to --verified-placement-angles-deg",
    )
    parser.add_argument(
        "--placement-tcp-max-reach-mm",
        type=float,
        default=float("inf"),
        help="Reject a suction plan before pickup when its aligned D or rotation-safe TCP exceeds this base-origin radius",
    )
    parser.add_argument(
        "--placement-tcp-y-max-mm",
        type=float,
        default=float("inf"),
        help="Reject a suction plan before pickup when its aligned D or rotation-safe TCP Y exceeds this limit",
    )
    parser.add_argument(
        "--placement-rotation-score-weight",
        type=float,
        default=10.0,
        help="Plan-score penalty per degree of loaded rotation required at D",
    )
    parser.add_argument(
        "--max-placement-alignment-error-deg",
        type=float,
        default=5.0,
        help="Maximum undirected long-edge error allowed for a non-degraded placement",
    )
    parser.add_argument(
        "--allow-degraded-placement",
        action="store_true",
        help=(
            "Allow release at a verified D fallback whose long-edge error exceeds the configured "
            "limit; the run is explicitly logged as degraded"
        ),
    )
    parser.add_argument(
        "--barcode-reader-bind-ip",
        default="192.168.2.100",
        help="Local TCP server address configured in the top/front barcode reader",
    )
    parser.add_argument(
        "--barcode-reader-port",
        type=int,
        default=3001,
        help="Local TCP server port configured in the top/front barcode reader",
    )
    parser.add_argument(
        "--barcode-reader-timeout-s",
        type=float,
        default=3.0,
        help="Maximum time to listen for top/front barcode data after the D-point delay",
    )
    parser.add_argument(
        "--disable-barcode-reader",
        action="store_true",
        help="Skip the top/front TCP barcode-reader step after placement at D",
    )
    parser.add_argument(
        "--suction-do-board",
        type=int,
        default=3,
        help="xCore IO board number for the suction valve output, e.g. DO3_5 uses board 5",
    )
    parser.add_argument(
        "--suction-do-port",
        type=int,
        default=6,
        help="Primary suction xCore DO port; physical mapping is DO3_6",
    )
    parser.add_argument(
        "--secondary-suction-do-port",
        type=int,
        default=5,
        help="Secondary suction xCore DO port; physical mapping is DO3_5",
    )
    parser.add_argument(
        "--secondary-suction-offset-y-mm",
        type=float,
        default=-245.0,
        help="Second cup center relative to the active TCP along tool Y; default is -245 mm",
    )
    parser.add_argument(
        "--disable-secondary-suction",
        action="store_true",
        help="Exclude the parallel secondary suction cup from automatic selection",
    )
    parser.add_argument(
        "--third-suction-do-port",
        type=int,
        default=4,
        help="xCore DO port for the perpendicular third suction cup; default is DO3_4",
    )
    parser.add_argument(
        "--third-suction-offset-xyz-mm",
        type=float,
        nargs=3,
        default=(102.5, 0.0, 113.0),
        metavar=("X", "Y", "Z"),
        help="Third cup center relative to the active/main TCP; default is 102.5 0 113 mm",
    )
    parser.add_argument(
        "--disable-third-suction",
        action="store_true",
        help="Exclude the perpendicular third suction cup from automatic selection",
    )
    parser.add_argument(
        "--fourth-suction-do-port",
        type=int,
        default=3,
        help="xCore DO port for the perpendicular fourth suction cup; default is DO3_3",
    )
    parser.add_argument(
        "--fourth-suction-offset-xyz-mm",
        type=float,
        nargs=3,
        default=(102.5, -245.0, 113.0),
        metavar=("X", "Y", "Z"),
        help="Fourth cup center relative to the active/main TCP; default is 102.5 -245 113 mm",
    )
    parser.add_argument(
        "--disable-fourth-suction",
        action="store_true",
        help="Exclude the perpendicular fourth suction cup from automatic selection",
    )
    parser.add_argument(
        "--force-suction-cups",
        nargs="+",
        choices=("1", "2", "3", "4"),
        default=None,
        metavar="CUP",
        help="Restrict pickup planning to specified physical cup numbers, e.g. --force-suction-cups 3 4",
    )
    parser.add_argument(
        "--left-zone-suction-cups",
        nargs="+",
        choices=("1", "2", "3", "4"),
        default=("1", "3"),
        metavar="CUP",
        help="Cups allowed when the grasp center is inside the left suction ROI; default: 1 3",
    )
    parser.add_argument(
        "--right-zone-suction-cups",
        nargs="+",
        choices=("1", "2", "3", "4"),
        default=("2", "4"),
        metavar="CUP",
        help="Cups allowed when the grasp center is inside the right suction ROI; default: 2 4",
    )
    parser.add_argument(
        "--dual-suction-clearance-mm",
        type=float,
        default=120.0,
        help="Preferred edge-to-edge clearance between any unused cup and a detected package",
    )
    parser.add_argument(
        "--unused-cup-min-clearance-mm",
        type=float,
        default=20.0,
        help="Hard minimum edge clearance; pickup plans below it are rejected",
    )
    parser.add_argument(
        "--suction-cup-collision-radius-mm",
        type=float,
        default=35.0,
        help="Collision radius around each unused suction cup, including its holder",
    )
    parser.add_argument(
        "--fixed-x-fallback-yaw-step-deg",
        type=float,
        default=30.0,
        help="Yaw step used only after the fixed tool-X grasp pose has no IK solution",
    )
    parser.add_argument(
        "--fixed-x-fallback-yaw-max-deg",
        type=float,
        default=90.0,
        help="Maximum yaw change allowed for fixed-X IK fallback while keeping the cup face down",
    )
    parser.add_argument(
        "--disable-fixed-x-yaw-fallback",
        action="store_true",
        help="Never rotate about the down-facing suction axis when the fixed-X pose has no IK",
    )
    parser.add_argument(
        "--disable-suction-io",
        action="store_true",
        help="Do not control the suction DO output; useful for dry mechanical tests",
    )
    parser.add_argument(
        "--invert-suction-io",
        action="store_true",
        help="Invert suction logic so a suction-on command writes the DO output OFF",
    )
    parser.add_argument(
        "--acknowledge-verified-tcp",
        action="store_true",
        help="Acknowledge that the active TCP and collision-free poses were physically verified at low speed",
    )
    parser.add_argument(
        "--suction-off-on-error",
        action="store_true",
        help="Turn suction DO off after a batch error before reaching D; default keeps it on for manual recovery",
    )
    parser.add_argument(
        "--pass-through-zone-mm",
        type=float,
        default=30.0,
        help="Blend radius for pass-through points A, A*, and B after pickup; C and D remain stop points",
    )
    parser.add_argument(
        "--analysis-refresh-s",
        type=float,
        default=2.0,
        help="Automatically refresh package recognition every N seconds while idle; set 0 to keep the old static result",
    )
    parser.add_argument(
        "--auto-cycle-settle-s",
        type=float,
        default=1.0,
        help="Delay after each completed batch before an automatic refreshed detection may start the next batch",
    )
    parser.add_argument(
        "--post-batch-discard-frames",
        type=int,
        default=5,
        help="Camera frames discarded after each batch so automatic mode does not consume stale queued images",
    )
    parser.add_argument("--robot-speed-mm-s", type=float, default=150.0, help="Robot speed")
    parser.add_argument("--robot-zone-mm", type=float, default=0.0, help="Robot zone/blend radius")
    parser.add_argument("--robot-timeout-s", type=float, default=60.0, help="Robot move timeout")
    return parser


def load_camera_matrix(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    return np.asarray(payload["camera_matrix"], dtype=np.float64)


def load_camera_point_to_base_transform(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    # In this project, the matrix stored under "base_to_camera" is the one that
    # maps camera points into the robot base frame.
    if "base_to_camera" in payload:
        block = payload["base_to_camera"]
    elif "camera_to_base" in payload:
        block = payload["camera_to_base"]
    else:
        raise KeyError("No hand-eye transform was found in the JSON file.")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(block["rotation_matrix"], dtype=np.float64)
    transform[:3, 3] = np.asarray(block["translation_m"], dtype=np.float64).reshape(3)
    return transform


def polygon_from_payload(payload: object) -> np.ndarray | None:
    points = np.asarray(payload if payload is not None else [], dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        return None
    return points


def load_roi_config(path: str | Path) -> RoiConfig:
    roi_path = Path(path)
    if not roi_path.exists():
        return RoiConfig(overall_polygon=None, support_polygons={}, exclude_polygons=[])

    payload = load_json(roi_path)
    if "polygon_pixels" in payload:
        overall_polygon = polygon_from_payload(payload.get("polygon_pixels"))
        return RoiConfig(overall_polygon=overall_polygon, support_polygons={}, exclude_polygons=[])

    overall_polygon = polygon_from_payload(payload.get("overall_polygon_pixels"))
    support_polygons: dict[str, np.ndarray] = {}
    support_block = payload.get("support_polygons", {})
    if isinstance(support_block, dict):
        for spec in SUPPORT_REGION_SPECS:
            polygon = polygon_from_payload(support_block.get(spec.region_id))
            if polygon is not None:
                support_polygons[spec.region_id] = polygon

    exclude_polygons: list[np.ndarray] = []
    for item in payload.get("exclude_polygons_pixels", []) or []:
        polygon = polygon_from_payload(item)
        if polygon is not None:
            exclude_polygons.append(polygon)
    suction_zone_polygons: dict[str, np.ndarray] = {}
    suction_zone_block = payload.get("suction_zone_polygons", {})
    if isinstance(suction_zone_block, dict):
        for zone_name in ("left", "right"):
            polygon = polygon_from_payload(suction_zone_block.get(zone_name))
            if polygon is not None:
                suction_zone_polygons[zone_name] = polygon
    return RoiConfig(
        overall_polygon=overall_polygon,
        support_polygons=support_polygons,
        exclude_polygons=exclude_polygons,
        suction_zone_polygons=suction_zone_polygons,
    )


def save_roi_config(path: str | Path, config: RoiConfig) -> None:
    payload = {
        "overall_polygon_pixels": None if config.overall_polygon is None else config.overall_polygon.astype(int).tolist(),
        "support_polygons": {
            spec.region_id: (
                config.support_polygons[spec.region_id].astype(int).tolist()
                if spec.region_id in config.support_polygons
                else None
            )
            for spec in SUPPORT_REGION_SPECS
        },
        "exclude_polygons_pixels": [polygon.astype(int).tolist() for polygon in config.exclude_polygons],
        "suction_zone_polygons": {
            zone_name: (
                config.suction_zone_polygons[zone_name].astype(int).tolist()
                if zone_name in config.suction_zone_polygons
                else None
            )
            for zone_name in ("left", "right")
        },
    }
    save_json(path, payload)


def delete_roi_config(path: str | Path) -> None:
    roi_path = Path(path)
    if roi_path.exists():
        roi_path.unlink()


def sample_point_cloud(
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    sample_step: int,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_mm.shape
    ys = np.arange(0, height, sample_step, dtype=np.int32)
    xs = np.arange(0, width, sample_step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    sampled_depth = depth_mm[grid_y.ravel(), grid_x.ravel()]
    valid = (sampled_depth > min_depth_mm) & (sampled_depth < max_depth_mm)

    pixels = pixels[valid]
    sampled_depth = sampled_depth[valid]
    if len(pixels) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    points = pixel_to_camera_points(pixels, sampled_depth, camera_matrix)
    return pixels, points


def pixel_to_camera_points(
    pixels: np.ndarray,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    u = pixels[:, 0].astype(np.float64)
    v = pixels[:, 1].astype(np.float64)
    z = depth_mm.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack([x, y, z])


def camera_to_pixel(points_camera_mm: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = points_camera_mm[:, 0]
    y = points_camera_mm[:, 1]
    z = points_camera_mm[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.column_stack([u, v])


def fit_plane_ransac(points_mm: np.ndarray, threshold_mm: float, iterations: int) -> PlaneModel:
    if len(points_mm) < 3:
        raise RuntimeError("At least 3 points are required to fit a plane.")

    rng = np.random.default_rng(523)
    best_inliers = np.empty((0,), dtype=np.int32)
    best_normal = None
    best_d = None

    for _ in range(iterations):
        sample = points_mm[rng.choice(len(points_mm), size=3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = float(normal @ sample[0])
        distances = np.abs(points_mm @ normal - d)
        inliers = np.where(distances < threshold_mm)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_normal is None or len(best_inliers) < 3:
        raise RuntimeError("Failed to fit a dominant plane.")

    centroid, normal = fit_plane_svd(points_mm[best_inliers])
    if float(normal @ centroid) > 0.0:
        normal = -normal
    d = float(normal @ centroid)
    return PlaneModel(normal=normal, d=d, centroid=centroid, inlier_indices=best_inliers)


def fit_plane_svd(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points_mm.mean(axis=0)
    centered = points_mm - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def fit_top_plane_ransac_svd(points_mm: np.ndarray, args: argparse.Namespace) -> PlaneModel | None:
    """Find robust top-plane inliers with RANSAC, then refine them with SVD."""
    if len(points_mm) < args.min_top_points:
        return None
    try:
        plane = fit_plane_ransac(
            points_mm,
            float(args.top_plane_ransac_threshold_mm),
            int(args.top_plane_ransac_iterations),
        )
    except RuntimeError:
        return None
    if len(plane.inlier_indices) < args.min_top_points:
        return None
    return plane


def plane_signed_distance(points_mm: np.ndarray, plane: PlaneModel) -> np.ndarray:
    return points_mm @ plane.normal - plane.d


def euclidean_cluster(points_mm: np.ndarray, tolerance_mm: float) -> list[np.ndarray]:
    if len(points_mm) == 0:
        return []

    tree = cKDTree(points_mm)
    visited = np.zeros(len(points_mm), dtype=bool)
    clusters: list[np.ndarray] = []

    for start_idx in range(len(points_mm)):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        visited[start_idx] = True
        cluster = []

        while stack:
            idx = stack.pop()
            cluster.append(idx)
            for nb in tree.query_ball_point(points_mm[idx], tolerance_mm):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        clusters.append(np.asarray(cluster, dtype=np.int32))

    return clusters


def dense_height_map(
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    plane: PlaneModel,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_mm.shape
    grid_y, grid_x = np.indices((height, width), dtype=np.int32)
    valid_mask = (depth_mm > min_depth_mm) & (depth_mm < max_depth_mm)

    height_map = np.full((height, width), np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return valid_mask, height_map

    pixels = np.column_stack([grid_x[valid_mask], grid_y[valid_mask]])
    valid_depth = depth_mm[valid_mask]
    points = pixel_to_camera_points(pixels, valid_depth, camera_matrix)
    height_map[valid_mask] = plane_signed_distance(points, plane).astype(np.float32)
    return valid_mask, height_map


def build_tray_mask(
    valid_mask: np.ndarray,
    height_map: np.ndarray,
    plane_threshold_mm: float,
    tray_erode_px: int,
) -> np.ndarray:
    plane_mask = valid_mask & (np.abs(height_map) <= plane_threshold_mm * 1.5)
    if not np.any(plane_mask):
        return np.zeros_like(valid_mask, dtype=bool)

    plane_u8 = (plane_mask.astype(np.uint8)) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    plane_u8 = cv2.morphologyEx(plane_u8, cv2.MORPH_CLOSE, close_kernel)

    points_yx = np.column_stack(np.nonzero(plane_u8 > 0))
    if len(points_yx) < 3:
        return plane_u8 > 0

    hull_points = np.column_stack([points_yx[:, 1], points_yx[:, 0]]).astype(np.int32)
    hull = cv2.convexHull(hull_points)
    tray_mask = np.zeros_like(plane_u8)
    cv2.fillConvexPoly(tray_mask, hull, 255)

    if tray_erode_px > 0:
        size = max(1, tray_erode_px)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
        tray_mask = cv2.erode(tray_mask, erode_kernel)

    return tray_mask > 0


def get_support_region_spec(region_id: str) -> SupportRegionSpec | None:
    for spec in SUPPORT_REGION_SPECS:
        if spec.region_id == region_id:
            return spec
    return None


def polygon_from_mask(mask: np.ndarray) -> np.ndarray | None:
    points_yx = np.column_stack(np.nonzero(mask))
    if len(points_yx) < 3:
        return None
    points_xy = np.column_stack([points_yx[:, 1], points_yx[:, 0]]).astype(np.int32)
    return cv2.convexHull(points_xy).reshape(-1, 2)


def roi_mask_from_polygon(image_shape: tuple[int, int], polygon: np.ndarray | None) -> np.ndarray | None:
    if polygon is None or len(polygon) < 3:
        return None

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask > 0


def exclude_mask_from_config(image_shape: tuple[int, int], roi_config: RoiConfig) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in roi_config.exclude_polygons:
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask > 0


def active_roi_mask_from_config(
    image_shape: tuple[int, int],
    roi_config: RoiConfig,
    exclude_mask: np.ndarray | None = None,
    erode_px: int = 3,
) -> np.ndarray:
    overall_mask = roi_mask_from_polygon(image_shape, roi_config.overall_polygon)
    if overall_mask is None:
        overall_mask = np.ones(image_shape, dtype=bool)

    if erode_px > 0:
        roi_u8 = overall_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        overall_mask = cv2.erode(roi_u8, kernel) > 0

    if exclude_mask is not None and exclude_mask.shape == overall_mask.shape:
        overall_mask = overall_mask & ~exclude_mask
    return overall_mask


def clean_object_mask(mask: np.ndarray, close_px: int, open_px: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8)) * 255

    if close_px > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_px * 2 + 1, close_px * 2 + 1),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    if open_px > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_px * 2 + 1, open_px * 2 + 1),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    return mask_u8 > 0


def compute_height_gradient(height_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    filled = np.where(valid_mask, height_map, 0.0).astype(np.float32)
    blurred = cv2.GaussianBlur(filled, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    gradient[~valid_mask] = np.inf
    return gradient


def region_bbox_xyxy(pixels: np.ndarray) -> np.ndarray:
    mins = pixels.min(axis=0)
    maxs = pixels.max(axis=0)
    return np.asarray([mins[0], mins[1], maxs[0], maxs[1]], dtype=np.float64)


def bbox_gap_px(a: np.ndarray, b: np.ndarray) -> float:
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return float(np.hypot(dx, dy))


def oriented_box_from_pixels(pixels: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(pixels.astype(np.float32))
    box = cv2.boxPoints(rect)
    return np.round(box).astype(np.int32)


def resize_depth_to_color_if_needed(
    depth_mm: np.ndarray,
    depth_display: np.ndarray,
    color_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if depth_mm.shape[:2] == color_shape:
        return depth_mm, depth_display

    width = color_shape[1]
    height = color_shape[0]
    resized_depth = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    resized_display = cv2.normalize(
        np.where(resized_depth > 0.0, resized_depth, 0.0),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return resized_depth.astype(np.float32), resized_display


def rectangle_overlap_over_smaller(a: RgbRectRegion, b: RgbRectRegion) -> float:
    intersection = int(np.count_nonzero(a.mask & b.mask))
    smaller = max(1, min(int(np.count_nonzero(a.mask)), int(np.count_nonzero(b.mask))))
    return float(intersection / smaller)


def candidate_overlap_over_smaller(
    a: ClusterCandidate,
    b: ClusterCandidate,
    image_shape: tuple[int, int],
) -> float:
    mask_a = np.zeros(image_shape, dtype=np.uint8)
    mask_b = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, a.hull_pixels.astype(np.int32), 255)
    cv2.fillConvexPoly(mask_b, b.hull_pixels.astype(np.int32), 255)
    intersection = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    smaller = max(1, min(int(np.count_nonzero(mask_a)), int(np.count_nonzero(mask_b))))
    return float(intersection / smaller)


def candidate_overlap_geometry(
    a: ClusterCandidate,
    b: ClusterCandidate,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return intersection, union and both rasterized polygon areas."""
    mask_a = np.zeros(image_shape, dtype=np.uint8)
    mask_b = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, a.hull_pixels.astype(np.int32), 1)
    cv2.fillConvexPoly(mask_b, b.hull_pixels.astype(np.int32), 1)
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    intersection = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    union = area_a + area_b - intersection
    return intersection, union, area_a, area_b


def candidates_are_duplicates(
    a: ClusterCandidate,
    b: ClusterCandidate,
    image_shape: tuple[int, int],
    iou_threshold: float,
    min_area_ratio: float,
    max_center_distance: float,
) -> bool:
    """Suppress only near-identical detections, never size-different containment."""
    if a.class_id != b.class_id:
        return False
    intersection, union, area_a, area_b = candidate_overlap_geometry(a, b, image_shape)
    if area_a <= 0 or area_b <= 0:
        return False
    iou = intersection / max(1, union)
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    center_distance_px = float(
        np.linalg.norm(
            np.asarray(a.center_pixel, dtype=np.float64)
            - np.asarray(b.center_pixel, dtype=np.float64)
        )
    )
    mean_equivalent_size_px = 0.5 * (np.sqrt(area_a) + np.sqrt(area_b))
    normalized_center_distance = center_distance_px / max(1.0, mean_equivalent_size_px)
    return (
        iou >= iou_threshold
        and area_ratio >= min_area_ratio
        and normalized_center_distance <= max_center_distance
    )


def prune_duplicate_candidates(
    candidates: list[ClusterCandidate],
    image_shape: tuple[int, int],
    iou_threshold: float,
    min_area_ratio: float = 0.70,
    max_center_distance: float = 0.20,
) -> list[ClusterCandidate]:
    def score(candidate: ClusterCandidate) -> tuple[float, float, float, float, float]:
        return (
            1.0 if candidate.motion_safe else 0.0,
            candidate.confidence,
            candidate.foreground_ratio,
            -candidate.flatness_mm,
            float(candidate.point_count),
        )

    kept: list[ClusterCandidate] = []
    for candidate in sorted(candidates, key=score, reverse=True):
        if any(
            candidates_are_duplicates(
                candidate,
                existing,
                image_shape,
                iou_threshold,
                min_area_ratio,
                max_center_distance,
            )
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def candidate_covered_ratio(
    candidate: ClusterCandidate,
    upper_candidate: ClusterCandidate,
    image_shape: tuple[int, int],
) -> float:
    candidate_mask = np.zeros(image_shape, dtype=np.uint8)
    upper_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(candidate_mask, [candidate.hull_pixels.astype(np.int32)], 1)
    cv2.fillPoly(upper_mask, [upper_candidate.hull_pixels.astype(np.int32)], 1)
    candidate_area = max(1, int(np.count_nonzero(candidate_mask)))
    intersection = int(np.count_nonzero((candidate_mask > 0) & (upper_mask > 0)))
    return float(intersection / candidate_area)


def rank_candidates_for_next_pick(
    candidates: list[ClusterCandidate],
    image_shape: tuple[int, int],
    min_upper_height_mm: float = 10.0,
    covered_ratio_threshold: float = 0.10,
    max_occlusion_ratio: float = 0.40,
    height_weight: float = 0.35,
    occlusion_weight: float = 0.30,
    flatness_weight: float = 0.15,
    confidence_weight: float = 0.10,
    point_quality_weight: float = 0.10,
    nested_small_bonus: float = 0.05,
) -> tuple[list[ClusterCandidate], dict[int, tuple[bool, float, int | None]]]:
    """Rank candidates with hard safety checks followed by a normalized weighted score."""
    if not candidates:
        return [], {}
    weights = np.asarray(
        [height_weight, occlusion_weight, flatness_weight, confidence_weight, point_quality_weight],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Next-pick weights must be non-negative and have a positive sum.")
    weights /= float(weights.sum())
    max_occlusion_ratio = float(np.clip(max_occlusion_ratio, 0.0, 1.0))

    coverage: dict[int, tuple[bool, float, int | None]] = {}
    nested_small_priority: dict[int, bool] = {id(candidate): False for candidate in candidates}
    for candidate in candidates:
        max_ratio = 0.0
        blocker_index: int | None = None
        for upper in candidates:
            if upper is candidate:
                continue
            if upper.point_base_mm[2] < candidate.point_base_mm[2] + min_upper_height_mm:
                continue
            ratio = candidate_covered_ratio(candidate, upper, image_shape)
            if ratio > max_ratio:
                max_ratio = ratio
                blocker_index = upper.index
        coverage[id(candidate)] = (max_ratio >= covered_ratio_threshold, max_ratio, blocker_index)

    # If a substantially smaller detection is contained by a larger one, keep
    # both and prefer the smaller candidate when their fitted surfaces are too
    # close to establish a reliable vertical order. This protects thin parcels
    # while the larger candidate is recomputed after the first pick.
    for candidate in candidates:
        for larger in candidates:
            if candidate is larger:
                continue
            intersection, _union, candidate_area, larger_area = candidate_overlap_geometry(
                candidate,
                larger,
                image_shape,
            )
            if candidate_area <= 0 or candidate_area >= larger_area:
                continue
            contained_ratio = intersection / candidate_area
            area_ratio = candidate_area / larger_area
            height_delta_mm = float(candidate.point_base_mm[2] - larger.point_base_mm[2])
            if contained_ratio >= 0.65 and area_ratio < 0.70 and height_delta_mm >= -10.0:
                nested_small_priority[id(candidate)] = True
                break

    def normalized(values: list[float], *, lower_is_better: bool = False) -> dict[int, float]:
        finite = np.asarray(values, dtype=np.float64)
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high - low <= 1e-9:
            scores = np.ones(len(values), dtype=np.float64)
        else:
            scores = (finite - low) / (high - low)
            if lower_is_better:
                scores = 1.0 - scores
        return {id(candidate): float(score) for candidate, score in zip(candidates, scores)}

    height_scores = normalized([float(item.point_base_mm[2]) for item in candidates])
    flatness_scores = normalized([float(item.flatness_mm) for item in candidates], lower_is_better=True)

    for candidate in candidates:
        _covered, occlusion_ratio, blocker_index = coverage[id(candidate)]
        severe_occlusion = blocker_index is not None and occlusion_ratio >= max_occlusion_ratio
        if severe_occlusion:
            candidate.motion_safe = False
            occlusion_note = f"occluded={occlusion_ratio:.2f}_by#{blocker_index}"
            candidate.filter_note = (
                occlusion_note if candidate.filter_note == "ok" else f"{candidate.filter_note},{occlusion_note}"
            )

        components = np.asarray(
            [
                height_scores[id(candidate)],
                1.0 - float(np.clip(occlusion_ratio, 0.0, 1.0)),
                flatness_scores[id(candidate)],
                float(np.clip(candidate.confidence, 0.0, 1.0)),
                float(np.clip(candidate.foreground_ratio, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
        bonus = nested_small_bonus if nested_small_priority[id(candidate)] else 0.0
        candidate.selection_score = float(weights @ components + bonus)
        candidate.selection_note = (
            f"height={components[0]:.2f},visible={components[1]:.2f},"
            f"flatness={components[2]:.2f},confidence={components[3]:.2f},"
            f"points={components[4]:.2f},nested_bonus={bonus:.2f}"
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.motion_safe else 1,
            -float(item.selection_score),
            -float(item.point_base_mm[2]),
            -float(item.confidence),
        ),
    )
    return ranked, coverage


def rgb_rect_regions_from_mask(
    mask: np.ndarray,
    source: str,
    start_label: int,
    args: argparse.Namespace,
    use_component_points: bool = False,
    allowed_mask: np.ndarray | None = None,
) -> list[RgbRectRegion]:
    if allowed_mask is None:
        allowed_mask = mask

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[RgbRectRegion] = []
    label = start_label

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0.0:
            continue

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        short_side = min(width, height)
        long_side = max(width, height)
        rect_area = float(width * height)
        if rect_area <= 1.0:
            continue
        if rect_area < args.rgb_min_rect_area_px or rect_area > args.rgb_max_rect_area_px:
            continue
        if short_side < args.rgb_min_rect_side_px or long_side > args.rgb_max_rect_side_px:
            continue

        rectangularity = contour_area / rect_area
        if rectangularity < args.rgb_min_rectangularity:
            continue

        hull = np.round(cv2.boxPoints(rect)).astype(np.int32)
        rect_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(rect_mask, hull, 255)
        rect_mask_bool = rect_mask > 0
        rect_pixels = int(np.count_nonzero(rect_mask_bool))
        rect_outside_ratio = 1.0 - (int(np.count_nonzero(rect_mask_bool & allowed_mask)) / max(1, rect_pixels))
        if rect_outside_ratio > args.rgb_max_rect_outside_roi_ratio:
            continue

        rect_mask_bool &= allowed_mask
        point_mask = None
        if use_component_points:
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
            point_mask = (component_mask > 0) & allowed_mask
        label += 1
        regions.append(
            RgbRectRegion(
                label=label,
                source=source,
                contour_area_px=contour_area,
                rect_area_px=rect_area,
                rectangularity=float(rectangularity),
                center_pixel=np.asarray(rect[0], dtype=np.float64),
                hull_pixels=hull,
                mask=rect_mask_bool,
                point_mask=point_mask,
            )
        )

    return regions


def split_binary_mask_instances(
    mask: np.ndarray,
    min_area_px: int,
    split_peak_threshold_ratio: float,
    split_peak_min_distance_px: float,
) -> list[np.ndarray]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    instances: list[np.ndarray] = []

    for label in range(1, labels_count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue

        component_mask = labels == label
        if area_px < max(min_area_px * 3, 900):
            instances.append(component_mask)
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        crop = component_mask[y : y + h, x : x + w].astype(np.uint8) * 255
        distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        max_distance = float(distance.max())
        if max_distance < 4.0:
            instances.append(component_mask)
            continue

        peak_threshold = max_distance * split_peak_threshold_ratio
        peak_mask = (distance >= peak_threshold).astype(np.uint8)
        peak_count, peak_labels = cv2.connectedComponents(peak_mask, connectivity=8)
        peak_centers: list[np.ndarray] = []
        for peak_label in range(1, peak_count):
            peak_ys, peak_xs = np.nonzero(peak_labels == peak_label)
            if len(peak_xs) == 0:
                continue
            peak_centers.append(np.asarray([peak_xs.mean() + x, peak_ys.mean() + y], dtype=np.float64))

        if len(peak_centers) <= 1:
            instances.append(component_mask)
            continue

        filtered_centers: list[np.ndarray] = []
        for center in peak_centers:
            if all(float(np.linalg.norm(center - existing)) >= split_peak_min_distance_px for existing in filtered_centers):
                filtered_centers.append(center)
        if len(filtered_centers) <= 1:
            instances.append(component_mask)
            continue

        ys, xs = np.nonzero(component_mask)
        pixels = np.column_stack([xs, ys]).astype(np.float64)
        centers = np.vstack(filtered_centers)
        distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        added = 0
        for assignment_index in range(len(filtered_centers)):
            member = assignments == assignment_index
            if int(np.count_nonzero(member)) < min_area_px:
                continue
            instance = np.zeros(mask.shape, dtype=bool)
            member_pixels = pixels[member].astype(np.int32)
            instance[member_pixels[:, 1], member_pixels[:, 0]] = True
            instances.append(instance)
            added += 1
        if added == 0:
            instances.append(component_mask)

    return instances


def rgb_rect_regions_from_instance_masks(
    masks: list[np.ndarray],
    source: str,
    start_label: int,
    args: argparse.Namespace,
    use_component_points: bool = False,
    allowed_mask: np.ndarray | None = None,
) -> list[RgbRectRegion]:
    regions: list[RgbRectRegion] = []
    for index, mask in enumerate(masks):
        regions.extend(
            rgb_rect_regions_from_mask(
                mask,
                source,
                start_label + index * 100,
                args,
                use_component_points=use_component_points,
                allowed_mask=allowed_mask,
            )
        )
    return regions


def rgb_rect_regions_from_seed_color(
    seed_masks: list[np.ndarray],
    package_color_mask: np.ndarray,
    source: str,
    start_label: int,
    args: argparse.Namespace,
    allowed_mask: np.ndarray | None = None,
) -> list[RgbRectRegion]:
    height, width = package_color_mask.shape
    regions: list[RgbRectRegion] = []
    pad = max(0, int(args.seed_color_expand_px))
    close_px = max(0, int(args.seed_color_close_px))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px * 2 + 1, close_px * 2 + 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))

    for index, seed_mask in enumerate(seed_masks):
        ys, xs = np.nonzero(seed_mask)
        if len(xs) == 0:
            continue

        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(width - 1, int(xs.max()) + pad)
        y1 = min(height - 1, int(ys.max()) + pad)
        crop_seed = seed_mask[y0 : y1 + 1, x0 : x1 + 1]
        crop_seed_gate = cv2.dilate(crop_seed.astype(np.uint8) * 255, gate_kernel) > 0
        crop_color = (package_color_mask[y0 : y1 + 1, x0 : x1 + 1] & crop_seed_gate) | crop_seed
        crop_color_u8 = crop_color.astype(np.uint8) * 255
        if close_px > 0:
            crop_color_u8 = cv2.morphologyEx(crop_color_u8, cv2.MORPH_CLOSE, close_kernel)
        crop_color_u8 = cv2.morphologyEx(crop_color_u8, cv2.MORPH_OPEN, open_kernel)

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(crop_color_u8, connectivity=8)
        selected_crop = np.zeros(crop_seed.shape, dtype=bool)
        min_overlap_px = max(12, int(np.count_nonzero(crop_seed) * 0.02))
        for label in range(1, labels_count):
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            if component_area < args.min_component_area_px:
                continue
            if int(np.count_nonzero(component & crop_seed)) >= min_overlap_px:
                selected_crop |= component

        if int(np.count_nonzero(selected_crop)) < args.min_component_area_px:
            selected_crop = crop_seed

        full_mask = np.zeros_like(package_color_mask, dtype=bool)
        full_mask[y0 : y1 + 1, x0 : x1 + 1] = selected_crop
        if allowed_mask is not None:
            full_mask &= allowed_mask
        regions.extend(
            rgb_rect_regions_from_mask(
                full_mask,
                source,
                start_label + index * 100,
                args,
                use_component_points=True,
                allowed_mask=allowed_mask,
            )
        )

    return regions


def make_mask_panel(mask: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.dtype == bool:
        panel = mask.astype(np.uint8) * 255
    else:
        panel = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel_bgr = cv2.resize(panel_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel_bgr, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return panel_bgr


def make_image_panel(image_bgr: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    panel = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return panel


def build_debug_montage(
    color_bgr: np.ndarray,
    debug_masks: dict[str, np.ndarray],
    rgb_regions: list[RgbRectRegion],
    candidates: list[ClusterCandidate],
) -> np.ndarray:
    candidate_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for region in rgb_regions:
        cv2.polylines(candidate_mask, [region.hull_pixels.astype(np.int32)], True, 180, 2)
    for candidate in candidates:
        cv2.polylines(candidate_mask, [candidate.hull_pixels.astype(np.int32)], True, 255, 3)

    panels = [
        make_image_panel(color_bgr, "color", (420, 260)),
        make_mask_panel(debug_masks.get("inner_roi", np.zeros(color_bgr.shape[:2], dtype=bool)), "inner_roi", (420, 260)),
        make_mask_panel(debug_masks.get("package_color_raw", np.zeros(color_bgr.shape[:2], dtype=bool)), "package_color_raw", (420, 260)),
        make_mask_panel(debug_masks.get("cardboard", np.zeros(color_bgr.shape[:2], dtype=bool)), "cardboard", (420, 260)),
        make_mask_panel(debug_masks.get("color_depth", np.zeros(color_bgr.shape[:2], dtype=bool)), "color_depth", (420, 260)),
        make_mask_panel(debug_masks.get("depth_foreground", np.zeros(color_bgr.shape[:2], dtype=bool)), "depth_foreground", (420, 260)),
        make_mask_panel(debug_masks.get("top_seed", np.zeros(color_bgr.shape[:2], dtype=bool)), "top_seed", (420, 260)),
        make_mask_panel(candidate_mask, "candidate_rects", (420, 260)),
    ]
    rows = [np.hstack(panels[index : index + 4]) for index in range(0, len(panels), 4)]
    return np.vstack(rows)


def load_yolo_detector(model_path: str | Path):
    config_dir = Path(model_path).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not available in the current virtual environment. "
            "Install ultralytics or run with --detector rgb."
        ) from exc
    return YOLO(str(model_path))


def tensor_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def yolo_class_name(yolo_model, class_id: int | None) -> str:
    if class_id is None:
        return "unknown"
    names = getattr(yolo_model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    fallback_names = {
        0: "parcel_box",
        1: "document_envelope",
        2: "soft_parcel",
    }
    return fallback_names.get(class_id, f"class_{class_id}")


def short_class_label(class_name: str) -> str:
    labels = {
        "parcel_box": "box",
        "document_envelope": "doc",
        "soft_parcel": "soft",
    }
    return labels.get(class_name, class_name[:10] if class_name else "unknown")


def yolo_polygon_region(
    polygon_xy: np.ndarray,
    source: str,
    label: int,
    class_id: int | None,
    class_name: str,
    confidence: float,
    image_shape: tuple[int, int],
    allowed_mask: np.ndarray,
) -> RgbRectRegion | None:
    polygon = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return None

    height, width = image_shape
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    polygon_i32 = np.round(polygon).astype(np.int32)

    mask_u8 = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask_u8, [polygon_i32], 255)
    mask = (mask_u8 > 0) & allowed_mask
    area_px = int(np.count_nonzero(mask))
    if area_px <= 0:
        return None

    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pixels)
    hull = np.round(cv2.boxPoints(rect)).astype(np.int32)
    rect_area = float(max(1.0, rect[1][0] * rect[1][1]))
    return RgbRectRegion(
        label=label,
        source=source,
        contour_area_px=float(area_px),
        rect_area_px=rect_area,
        rectangularity=float(min(1.0, area_px / rect_area)),
        center_pixel=np.asarray(rect[0], dtype=np.float64),
        hull_pixels=hull,
        mask=mask,
        point_mask=mask,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


def find_yolo_rectangle_regions(
    color_bgr: np.ndarray,
    roi_config: RoiConfig,
    args: argparse.Namespace,
    yolo_model,
    exclude_mask: np.ndarray | None = None,
) -> tuple[list[RgbRectRegion], list[str], dict[str, np.ndarray]]:
    image_shape = color_bgr.shape[:2]
    active_roi = active_roi_mask_from_config(image_shape, roi_config, exclude_mask, erode_px=3)
    result = yolo_model.predict(
        source=color_bgr,
        imgsz=args.yolo_imgsz,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        verbose=False,
    )[0]

    regions: list[RgbRectRegion] = []
    source_name = "yolo_unknown"
    raw_count = 0
    label = 80000

    if getattr(result, "obb", None) is not None and result.obb is not None and len(result.obb) > 0:
        source_name = "yolo_obb"
        polygons = tensor_to_numpy(result.obb.xyxyxyxy)
        confs = tensor_to_numpy(result.obb.conf)
        classes = tensor_to_numpy(getattr(result.obb, "cls", None))
        raw_count = len(polygons)
        for index, polygon in enumerate(polygons):
            confidence = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(classes[index]) if index < len(classes) else None
            region = yolo_polygon_region(
                polygon,
                source_name,
                label + index,
                class_id,
                yolo_class_name(yolo_model, class_id),
                confidence,
                image_shape,
                active_roi,
            )
            if region is not None:
                regions.append(region)
    elif getattr(result, "masks", None) is not None and result.masks is not None and result.masks.xy is not None:
        source_name = "yolo_seg"
        polygons = result.masks.xy
        confs = tensor_to_numpy(result.boxes.conf) if getattr(result, "boxes", None) is not None else np.empty((0,))
        classes = tensor_to_numpy(result.boxes.cls) if getattr(result, "boxes", None) is not None else np.empty((0,))
        raw_count = len(polygons)
        for index, polygon in enumerate(polygons):
            confidence = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(classes[index]) if index < len(classes) else None
            region = yolo_polygon_region(
                np.asarray(polygon),
                source_name,
                label + index,
                class_id,
                yolo_class_name(yolo_model, class_id),
                confidence,
                image_shape,
                active_roi,
            )
            if region is not None:
                regions.append(region)
    elif getattr(result, "boxes", None) is not None and result.boxes is not None and len(result.boxes) > 0:
        source_name = "yolo_box"
        boxes = tensor_to_numpy(result.boxes.xyxy)
        confs = tensor_to_numpy(result.boxes.conf)
        classes = tensor_to_numpy(result.boxes.cls)
        raw_count = len(boxes)
        for index, box in enumerate(boxes):
            x0, y0, x1, y1 = box.astype(float)
            polygon = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            confidence = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(classes[index]) if index < len(classes) else None
            region = yolo_polygon_region(
                polygon,
                source_name,
                label + index,
                class_id,
                yolo_class_name(yolo_model, class_id),
                confidence,
                image_shape,
                active_roi,
            )
            if region is not None:
                regions.append(region)

    # Ultralytics has already applied task-aware NMS (including rotated-box
    # NMS for OBB models). Do not apply containment/IoS suppression here:
    # a small thin package on a larger package is a valid nested detection.
    kept = regions

    yolo_mask = np.zeros(image_shape, dtype=bool)
    for region in kept:
        yolo_mask |= region.mask
    class_counts: dict[str, int] = {}
    for region in kept:
        class_counts[region.class_name] = class_counts.get(region.class_name, 0) + 1

    notes = [
        f"yolo_model={args.yolo_model}",
        f"yolo_task={getattr(yolo_model, 'task', 'unknown')}",
        f"yolo_source={source_name}",
        f"yolo_raw={raw_count}",
        f"yolo_kept={len(kept)}",
        f"yolo_mask_pixels={int(np.count_nonzero(yolo_mask))}",
        f"exclude_pixels={0 if exclude_mask is None else int(np.count_nonzero(exclude_mask))}",
    ]
    if class_counts:
        notes.append("yolo_classes=" + " ".join(f"{name}:{count}" for name, count in sorted(class_counts.items())))
    debug_masks = {
        "inner_roi": active_roi,
        "package_color_raw": yolo_mask,
        "cardboard": yolo_mask,
        "package_color": yolo_mask,
        "seed_color_source": yolo_mask,
        "color_depth": yolo_mask,
        "depth_foreground": yolo_mask,
        "edge": yolo_mask,
    }
    return kept, notes, debug_masks


def find_rgb_rectangle_regions(
    color_bgr: np.ndarray,
    roi_config: RoiConfig,
    args: argparse.Namespace,
    depth_hint_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[list[RgbRectRegion], list[str], dict[str, np.ndarray]]:
    image_shape = color_bgr.shape[:2]
    overall_mask = roi_mask_from_polygon(image_shape, roi_config.overall_polygon)
    if overall_mask is None:
        overall_mask = np.ones(image_shape, dtype=bool)

    roi_u8 = overall_mask.astype(np.uint8) * 255
    inner_roi_u8 = cv2.erode(roi_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    inner_roi = inner_roi_u8 > 0
    if exclude_mask is not None and exclude_mask.shape == inner_roi.shape:
        inner_roi = inner_roi & ~exclude_mask

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(gray_blur, args.rgb_canny_low, args.rgb_canny_high)
    edges[~inner_roi] = 0
    if args.rgb_edge_dilate_px > 0:
        size = args.rgb_edge_dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        edges = cv2.dilate(edges, kernel)

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    color_mask = (saturation >= args.rgb_min_saturation) & (value > 35) & inner_roi

    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_sample_mask = (
        inner_roi
        & (saturation <= args.rgb_background_max_saturation)
        & (value > 45)
    )
    if int(np.count_nonzero(background_sample_mask)) < 300:
        background_sample_mask = inner_roi & (value > 45)
    if int(np.count_nonzero(background_sample_mask)) > 0:
        background_lab = np.median(lab[background_sample_mask], axis=0)
    else:
        background_lab = np.asarray([128.0, 128.0, 128.0], dtype=np.float32)
    lab_distance = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
    color_distance_mask = (
        (lab_distance >= args.rgb_lab_distance)
        & (value > 35)
        & inner_roi
    )
    cardboard_mask = (
        (hue >= args.rgb_cardboard_hue_min)
        & (hue <= args.rgb_cardboard_hue_max)
        & (saturation >= max(8, args.rgb_min_saturation // 2))
        & (value > 35)
        & inner_roi
    )
    cardboard_strict_mask = (
        (hue >= args.rgb_cardboard_hue_min)
        & (hue <= args.rgb_cardboard_hue_max)
        & (saturation >= args.rgb_cardboard_strict_saturation)
        & (value >= args.rgb_cardboard_value_min)
        & (value <= args.rgb_cardboard_value_max)
        & inner_roi
    )
    package_color_mask = color_distance_mask | cardboard_mask

    close_size = args.rgb_edge_close_px * 2 + 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    color_close_size = args.rgb_color_close_px * 2 + 1
    color_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (color_close_size, color_close_size))
    color_depth_close_size = args.rgb_color_depth_close_px * 2 + 1
    color_depth_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (color_depth_close_size, color_depth_close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel) > 0
    edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, open_kernel) > 0
    color_mask = cv2.morphologyEx(color_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kernel) > 0
    color_mask = cv2.morphologyEx(color_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, open_kernel) > 0
    package_color_raw_mask = package_color_mask
    package_color_mask = cv2.morphologyEx(
        package_color_raw_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        color_close_kernel,
    ) > 0
    package_color_mask = cv2.morphologyEx(
        package_color_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        open_kernel,
    ) > 0
    cardboard_strict_mask = cv2.morphologyEx(
        cardboard_strict_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        color_depth_close_kernel,
    ) > 0
    cardboard_strict_mask = cv2.morphologyEx(
        cardboard_strict_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        open_kernel,
    ) > 0
    if depth_hint_mask is not None and depth_hint_mask.shape == package_color_mask.shape:
        depth_mask = depth_hint_mask & inner_roi
        color_depth_mask = package_color_raw_mask & depth_mask
        color_depth_mask = cv2.morphologyEx(
            color_depth_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            color_depth_close_kernel,
        ) > 0
        color_depth_mask = cv2.morphologyEx(
            color_depth_mask.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            open_kernel,
        ) > 0
    else:
        depth_mask = np.zeros_like(package_color_mask, dtype=bool)
        color_depth_mask = np.zeros_like(package_color_mask, dtype=bool)

    instance_min_area = int(max(args.min_component_area_px, args.rgb_min_rect_area_px * 0.25))
    color_depth_instances = split_binary_mask_instances(
        color_depth_mask,
        instance_min_area,
        args.split_peak_threshold_ratio,
        args.split_peak_min_distance_px,
    )
    depth_instances = split_binary_mask_instances(
        depth_mask,
        instance_min_area,
        args.split_peak_threshold_ratio,
        args.split_peak_min_distance_px,
    )
    cardboard_instances = split_binary_mask_instances(
        cardboard_strict_mask,
        instance_min_area,
        args.split_peak_threshold_ratio,
        args.split_peak_min_distance_px,
    )

    seed_color_source_mask = package_color_raw_mask | color_depth_mask
    candidates: list[RgbRectRegion] = []
    if not args.disable_cardboard_candidates:
        candidates.extend(
            rgb_rect_regions_from_instance_masks(
                cardboard_instances,
                "cardboard",
                70000,
                args,
                use_component_points=True,
                allowed_mask=inner_roi,
            )
        )
    candidates.extend(
        rgb_rect_regions_from_seed_color(
            color_depth_instances,
            seed_color_source_mask,
            "seed_color",
            50000,
            args,
            allowed_mask=inner_roi,
        )
    )
    candidates.extend(
        rgb_rect_regions_from_seed_color(
            depth_instances,
            seed_color_source_mask,
            "depth_seed_color",
            60000,
            args,
            allowed_mask=inner_roi,
        )
    )
    candidates.extend(
        rgb_rect_regions_from_instance_masks(
            color_depth_instances,
            "color_depth",
            30000,
            args,
            use_component_points=True,
            allowed_mask=inner_roi,
        )
    )
    candidates.extend(
        rgb_rect_regions_from_instance_masks(
            depth_instances,
            "depth_instance",
            40000,
            args,
            use_component_points=True,
            allowed_mask=inner_roi,
        )
    )
    if args.enable_edge_candidates:
        candidates.extend(rgb_rect_regions_from_mask(edge_mask, "edge", 0, args, allowed_mask=inner_roi))
    if args.enable_color_only_candidates:
        candidates.extend(
            rgb_rect_regions_from_mask(
                color_mask,
                "color",
                10000,
                args,
                use_component_points=True,
                allowed_mask=inner_roi,
            )
        )
    if args.enable_package_color_candidates:
        candidates.extend(
            rgb_rect_regions_from_mask(
                package_color_mask,
                "package_color",
                20000,
                args,
                use_component_points=True,
                allowed_mask=inner_roi,
            )
        )
    source_priority = {
        "cardboard": 7,
        "seed_color": 6,
        "depth_seed_color": 5,
        "color_depth": 4,
        "depth_instance": 3,
        "package_color": 2,
        "color": 2,
        "edge": 1,
    }
    candidates.sort(
        key=lambda item: (
            source_priority.get(item.source, 0),
            item.rectangularity,
            -abs(item.rect_area_px - 18000.0),
        ),
        reverse=True,
    )

    kept: list[RgbRectRegion] = []
    for candidate in candidates:
        if any(rectangle_overlap_over_smaller(candidate, existing) >= args.rgb_nms_overlap for existing in kept):
            continue
        kept.append(candidate)
    source_counts: dict[str, int] = {}
    for candidate in kept:
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1

    notes = [
        f"rgb_rect_raw={len(candidates)}",
        f"rgb_rect_kept={len(kept)}",
        "rgb_sources=" + " ".join(f"{source}:{count}" for source, count in sorted(source_counts.items())),
        f"rgb_bg_lab={background_lab.round(1).tolist()}",
        f"rgb_color_pixels={int(np.count_nonzero(package_color_mask))}",
        f"cardboard_pixels={int(np.count_nonzero(cardboard_strict_mask))}",
        f"rgb_color_depth_pixels={int(np.count_nonzero(color_depth_mask))}",
        (
            f"rgb_color_depth_instances={len(color_depth_instances)} "
            f"depth_instances={len(depth_instances)} "
            f"cardboard_instances={len(cardboard_instances)}"
        ),
        f"rgb_seed_source_pixels={int(np.count_nonzero(seed_color_source_mask))}",
        f"exclude_pixels={0 if exclude_mask is None else int(np.count_nonzero(exclude_mask))}",
    ]
    debug_masks = {
        "inner_roi": inner_roi,
        "package_color_raw": package_color_raw_mask,
        "cardboard": cardboard_strict_mask,
        "package_color": package_color_mask,
        "seed_color_source": seed_color_source_mask,
        "color_depth": color_depth_mask,
        "depth_foreground": depth_mask,
        "edge": edge_mask,
    }
    return kept, notes, debug_masks


def split_seed_component(
    component_mask: np.ndarray,
    component_pixels: np.ndarray,
    component_points: np.ndarray,
    component_heights: np.ndarray,
    label: int,
    min_seed_area_px: int,
    split_peak_threshold_ratio: float,
    split_peak_min_distance_px: float,
) -> list[SeedRegion]:
    area_px = len(component_pixels)
    bbox = region_bbox_xyxy(component_pixels)
    region = SeedRegion(
        label=label,
        area_px=area_px,
        center_pixel=component_pixels.mean(axis=0),
        top_height_mm=float(np.percentile(component_heights, 95)),
        bbox_xyxy=bbox,
        pixels=component_pixels,
        points=component_points,
        heights=component_heights,
    )
    if area_px < max(min_seed_area_px * 2, 120):
        return [region]

    x0, y0, x1, y1 = bbox.astype(int)
    crop = component_mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255
    distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance < 3.0:
        return [region]

    threshold = max_distance * split_peak_threshold_ratio
    peak_mask = (distance >= threshold).astype(np.uint8)
    peak_labels_count, peak_labels = cv2.connectedComponents(peak_mask, connectivity=8)
    if peak_labels_count <= 2:
        return [region]

    peak_centers: list[np.ndarray] = []
    for peak_label in range(1, peak_labels_count):
        peak_ys, peak_xs = np.nonzero(peak_labels == peak_label)
        if len(peak_xs) == 0:
            continue
        peak_centers.append(np.asarray([peak_xs.mean() + x0, peak_ys.mean() + y0], dtype=np.float64))

    if len(peak_centers) <= 1:
        return [region]

    min_peak_distance = min(
        float(np.linalg.norm(peak_centers[i] - peak_centers[j]))
        for i in range(len(peak_centers))
        for j in range(i + 1, len(peak_centers))
    )
    if min_peak_distance < split_peak_min_distance_px:
        return [region]

    centers = np.vstack(peak_centers)
    distances = np.linalg.norm(component_pixels[:, None, :].astype(np.float64) - centers[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)

    regions: list[SeedRegion] = []
    for assignment_idx in range(len(peak_centers)):
        member_mask = assignments == assignment_idx
        if int(np.count_nonzero(member_mask)) < min_seed_area_px:
            continue
        sub_pixels = component_pixels[member_mask]
        sub_points = component_points[member_mask]
        sub_heights = component_heights[member_mask]
        regions.append(
            SeedRegion(
                label=label * 100 + assignment_idx,
                area_px=len(sub_pixels),
                center_pixel=sub_pixels.mean(axis=0),
                top_height_mm=float(np.percentile(sub_heights, 95)),
                bbox_xyxy=region_bbox_xyxy(sub_pixels),
                pixels=sub_pixels,
                points=sub_points,
                heights=sub_heights,
            )
        )

    return regions if regions else [region]


def merge_seed_regions(
    seed_regions: list[SeedRegion],
    merge_distance_px: float,
    merge_height_mm: float,
    merge_gap_px: float,
) -> list[SeedRegion]:
    if not seed_regions:
        return []

    parent = list(range(len(seed_regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(seed_regions)):
        for j in range(i + 1, len(seed_regions)):
            pixel_distance = float(np.linalg.norm(seed_regions[i].center_pixel - seed_regions[j].center_pixel))
            height_distance = abs(seed_regions[i].top_height_mm - seed_regions[j].top_height_mm)
            gap_distance = bbox_gap_px(seed_regions[i].bbox_xyxy, seed_regions[j].bbox_xyxy)
            if (
                pixel_distance <= merge_distance_px
                and height_distance <= merge_height_mm
                and gap_distance <= merge_gap_px
            ):
                union(i, j)

    grouped: dict[int, list[SeedRegion]] = {}
    for index, region in enumerate(seed_regions):
        grouped.setdefault(find(index), []).append(region)

    merged_regions: list[SeedRegion] = []
    for group in grouped.values():
        pixels = np.vstack([item.pixels for item in group]).astype(np.int32)
        points = np.vstack([item.points for item in group]).astype(np.float64)
        heights = np.concatenate([item.heights for item in group]).astype(np.float64)
        center_pixel = pixels.mean(axis=0)
        merged_regions.append(
            SeedRegion(
                label=group[0].label,
                area_px=len(pixels),
                center_pixel=center_pixel,
                top_height_mm=float(np.percentile(heights, 95)),
                bbox_xyxy=region_bbox_xyxy(pixels),
                pixels=pixels,
                points=points,
                heights=heights,
            )
        )

    merged_regions.sort(key=lambda item: item.area_px, reverse=True)
    return merged_regions


def camera_point_to_base_mm(transform: np.ndarray, point_camera_mm: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point_camera_mm / 1000.0
    point_base_m = transform @ homogeneous
    return point_base_m[:3] * 1000.0


def normal_camera_to_base(transform: np.ndarray, normal_camera: np.ndarray) -> np.ndarray:
    normal_base = transform[:3, :3] @ normal_camera
    norm = np.linalg.norm(normal_base)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal_base / norm


def normal_base_to_camera(transform: np.ndarray, normal_base: np.ndarray) -> np.ndarray:
    normal_camera = transform[:3, :3].T @ normal_base
    norm = np.linalg.norm(normal_camera)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal_camera / norm


def ray_plane_intersection_from_pixel(
    pixel_xy: np.ndarray,
    camera_matrix: np.ndarray,
    plane_point_mm: np.ndarray,
    plane_normal: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
) -> np.ndarray | None:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    u = float(pixel_xy[0])
    v = float(pixel_xy[1])
    ray = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    denominator = float(plane_normal @ ray)
    if abs(denominator) < 1e-9:
        return None
    depth_mm = float((plane_normal @ plane_point_mm) / denominator)
    if not (min_depth_mm <= depth_mm <= max_depth_mm):
        return None
    return ray * depth_mm


def rectangle_short_axis_pixels(hull_pixels: np.ndarray) -> np.ndarray | None:
    corners = np.asarray(hull_pixels, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 4:
        return None
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if not np.all(np.isfinite(lengths)) or float(np.max(lengths)) < 1e-6:
        return None
    edge_index = int(np.argmin(lengths))
    short_axis = edges[edge_index]
    short_length = float(np.linalg.norm(short_axis))
    if short_length < 1e-6:
        return None
    return short_axis / short_length


def short_axis_base_from_rectangle(
    hull_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    plane_point_camera_mm: np.ndarray,
    plane_normal_camera: np.ndarray,
    camera_point_to_base: np.ndarray,
    normal_base: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    short_axis_pixels = rectangle_short_axis_pixels(hull_pixels)
    if short_axis_pixels is None:
        return None, None

    corners = np.asarray(hull_pixels, dtype=np.float64).reshape(-1, 2)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    edge_index = int(np.argmin(lengths))
    pixel_start = corners[edge_index]
    pixel_end = corners[(edge_index + 1) % len(corners)]
    point_start_camera = ray_plane_intersection_from_pixel(
        pixel_start,
        camera_matrix,
        plane_point_camera_mm,
        plane_normal_camera,
        min_depth_mm,
        max_depth_mm,
    )
    point_end_camera = ray_plane_intersection_from_pixel(
        pixel_end,
        camera_matrix,
        plane_point_camera_mm,
        plane_normal_camera,
        min_depth_mm,
        max_depth_mm,
    )
    if point_start_camera is None or point_end_camera is None:
        return short_axis_pixels, None

    short_axis_camera = point_end_camera - point_start_camera
    camera_length = float(np.linalg.norm(short_axis_camera))
    if camera_length < 1e-6:
        return short_axis_pixels, None
    short_axis_camera /= camera_length

    short_axis_base = camera_point_to_base[:3, :3] @ short_axis_camera
    short_axis_base -= normal_base * float(np.dot(short_axis_base, normal_base))
    base_length = float(np.linalg.norm(short_axis_base))
    if base_length < 1e-6:
        return short_axis_pixels, None
    short_axis_base /= base_length
    return short_axis_camera, short_axis_base


def select_grasp_normal(
    top_normal_camera: np.ndarray,
    support_normal_camera: np.ndarray,
    camera_point_to_base: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str]:
    base_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    top_normal_base = normal_camera_to_base(camera_point_to_base, top_normal_camera)
    support_normal_base = normal_camera_to_base(camera_point_to_base, support_normal_camera)

    if top_normal_base[2] < 0.0:
        top_normal_base = -top_normal_base
        top_normal_camera = -top_normal_camera
    if support_normal_base[2] < 0.0:
        support_normal_base = -support_normal_base
        support_normal_camera = -support_normal_camera

    mode = args.normal_mode
    if mode == "base-z":
        selected_base = base_z
        selected_mode = "base_z"
    elif mode == "support-plane":
        selected_base = support_normal_base
        selected_mode = "support_plane"
    elif mode == "top-plane":
        selected_base = top_normal_base
        selected_mode = "top_plane"
    else:
        snap_cos = float(np.cos(np.radians(args.normal_snap_angle_deg)))
        if top_normal_base[2] >= snap_cos:
            selected_base = base_z
            selected_mode = "auto_base_z"
        else:
            selected_base = top_normal_base
            selected_mode = "auto_top_plane"

    selected_base = selected_base / max(1e-9, np.linalg.norm(selected_base))
    selected_camera = normal_base_to_camera(camera_point_to_base, selected_base)
    return selected_camera, selected_base, selected_mode


def point_in_workspace(point_base_mm: np.ndarray, args: argparse.Namespace) -> bool:
    return bool(
        args.workspace_x_min_mm <= point_base_mm[0] <= args.workspace_x_max_mm
        and args.workspace_y_min_mm <= point_base_mm[1] <= args.workspace_y_max_mm
        and args.workspace_z_min_mm <= point_base_mm[2] <= args.workspace_z_max_mm
    )


def target_point_for_candidate(
    surface_base_mm: np.ndarray,
    normal_base: np.ndarray,
    standoff_mm: float,
    standoff_mode: str,
) -> np.ndarray:
    if standoff_mode == "normal":
        return surface_base_mm + standoff_mm * normal_base
    return surface_base_mm + np.asarray([0.0, 0.0, standoff_mm], dtype=np.float64)


def candidate_in_workspace(
    surface_base_mm: np.ndarray,
    normal_base: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    target_base_mm = target_point_for_candidate(
        surface_base_mm,
        normal_base,
        args.standoff_mm,
        args.standoff_mode,
    )
    normal = np.asarray(normal_base, dtype=np.float64)
    normal_length = float(np.linalg.norm(normal))
    if normal_length < 1e-9:
        return False
    pickup_base_mm = target_base_mm - float(args.pickup_down_mm) * (normal / normal_length)
    return (
        point_in_workspace(surface_base_mm, args)
        and point_in_workspace(target_base_mm, args)
        and point_in_workspace(pickup_base_mm, args)
    )


def target_rpy_for_candidate(
    candidate: ClusterCandidate,
    current_pose,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if args.rpy_mode == "fixed":
        if args.fixed_pickup_rpy_deg is None:
            raise RuntimeError("fixed RPY mode requires --fixed-pickup-rpy-deg")
        return np.asarray(args.fixed_pickup_rpy_deg, dtype=np.float64), "fixed"
    if args.align_normal_rpy or args.rpy_mode == "align-normal":
        placement_candidates = placement_aligned_pickup_rpy_candidates(
            candidate,
            current_pose,
            args,
        )
        if placement_candidates:
            placement_mode, placement_rpy_deg = placement_candidates[0]
            return placement_rpy_deg, placement_mode
        return (
            compute_grasp_rpy_from_normal_and_fixed_x(
                candidate.normal_base,
                FIXED_GRASP_X_YAW_DEG,
                args.tool_contact_axis,
            ),
            "align_normal_fixed_x",
        )
    if args.rpy_mode == "keep-current":
        return current_pose.rpy_deg_xyz(), "keep_current"

    current_rpy_deg = current_pose.rpy_deg_xyz()
    rz_deg = current_rpy_deg[2] if args.vertical_rz_deg is None else args.vertical_rz_deg
    return (
        np.asarray([args.vertical_rx_deg, args.vertical_ry_deg, rz_deg], dtype=np.float64),
        "vertical_down",
    )


def placement_aligned_pickup_rpy_candidates(
    candidate: ClusterCandidate,
    current_pose,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    """Orient the empty tool so D needs no loaded package rotation."""
    try:
        long_axis_base = package_long_axis_base(candidate)
    except Exception:
        return []

    package_long_yaw_deg = float(
        np.degrees(np.arctan2(long_axis_base[1], long_axis_base[0]))
    )
    # Tool Y must be parallel to the package long edge at pickup. The two
    # directions are equivalent because a rectangle long axis is undirected.
    tool_x_yaws_deg = [package_long_yaw_deg - 90.0, package_long_yaw_deg + 90.0]
    current_rotation = rpy_xyz_to_matrix(current_pose.rpy_rad_xyz)
    candidates: list[tuple[float, str, np.ndarray]] = []
    for tool_x_yaw_deg in tool_x_yaws_deg:
        wrapped_yaw_deg = (tool_x_yaw_deg + 180.0) % 360.0 - 180.0
        rpy_deg = compute_grasp_rpy_from_normal_and_fixed_x(
            candidate.normal_base,
            wrapped_yaw_deg,
            args.tool_contact_axis,
        )
        distance_deg = rotation_distance_deg(
            current_rotation,
            rpy_xyz_to_matrix(np.radians(rpy_deg)),
        )
        candidates.append(
            (
                distance_deg,
                f"align_normal_package_long_for_d_yaw_{wrapped_yaw_deg:+.1f}deg",
                rpy_deg,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [(mode, rpy_deg) for _distance, mode, rpy_deg in candidates]


def target_rpy_candidates_for_candidate(
    candidate: ClusterCandidate,
    current_pose,
    args: argparse.Namespace,
    primary_rpy_deg: np.ndarray,
    primary_rpy_mode: str,
) -> list[tuple[str, np.ndarray]]:
    if not (args.align_normal_rpy or args.rpy_mode == "align-normal"):
        return [(primary_rpy_mode, primary_rpy_deg)]
    if primary_rpy_mode == "align_normal_fixed_x" or primary_rpy_mode.startswith(
        "align_normal_package_long_for_d"
    ):
        candidates: list[tuple[str, np.ndarray]] = []
        if primary_rpy_mode.startswith("align_normal_package_long_for_d"):
            candidates.extend(
                placement_aligned_pickup_rpy_candidates(candidate, current_pose, args)
            )

        fixed_rpy_deg = compute_grasp_rpy_from_normal_and_fixed_x(
            candidate.normal_base,
            FIXED_GRASP_X_YAW_DEG,
            args.tool_contact_axis,
        )
        if not any(
            np.allclose(existing_rpy, fixed_rpy_deg, atol=1e-6)
            for _existing_mode, existing_rpy in candidates
        ):
            candidates.append(("align_normal_fixed_x", fixed_rpy_deg))
        if args.disable_fixed_x_yaw_fallback:
            return candidates

        step_deg = max(1.0, float(args.fixed_x_fallback_yaw_step_deg))
        max_deg = max(0.0, float(args.fixed_x_fallback_yaw_max_deg))
        offset_deg = step_deg
        while offset_deg <= max_deg + 1e-6:
            # Search both wrist-yaw directions. The previous positive-only
            # search could test -66/-36/-6 deg from a -96 deg fixed heading
            # while omitting the equally valid -126/-156/+174 deg branches.
            for signed_offset_deg in (offset_deg, -offset_deg):
                fallback_rpy_deg = compute_grasp_rpy_from_normal_and_fixed_x(
                    candidate.normal_base,
                    FIXED_GRASP_X_YAW_DEG + signed_offset_deg,
                    args.tool_contact_axis,
                )
                if not any(
                    np.allclose(existing_rpy, fallback_rpy_deg, atol=1e-6)
                    for _existing_mode, existing_rpy in candidates
                ):
                    candidates.append(
                        (
                            f"align_normal_fixed_x_fallback_yaw_{signed_offset_deg:+.1f}deg",
                            fallback_rpy_deg,
                        )
                    )
            offset_deg += step_deg
        return candidates
    no_flip_rpy_deg = compute_grasp_rpy_from_normal(
        candidate.normal_base,
        current_pose.rpy_rad_xyz,
        args.tool_contact_axis,
        args.tool_face_reference_rpy_deg,
    )
    candidates: list[tuple[str, np.ndarray]] = []
    if args.tool_y_mode != "obb-short-edge" or candidate.short_axis_base is None:
        # With the small suction cup, rotation about the vertical contact axis
        # does not change the down-facing contact direction.  A single
        # keep-current yaw can nevertheless put the wrist outside its IK/joint
        # limits, so search nearby yaw-equivalent poses before rejecting an
        # otherwise reachable XYZ target.
        # Site testing with suction cup 2 established -60 deg as the first
        # consistently solvable wrist branch for B -> loading-area travel.
        # Try that branch first to avoid repeatedly shuttling B <-> target XY.
        # A* is now the fixed, taught staging pose. Target-yaw alternatives are
        # therefore tried only from A*: a rejected A target does not start a
        # B <-> loading-area shuttle. Keep the previously successful -60 deg
        # branch first, then search other wrist branches for deep targets.
        yaw_offsets_deg = [-60.0, -90.0, -45.0, -30.0, 0.0, 30.0, 60.0, 90.0]
        for offset_deg in yaw_offsets_deg:
            rpy_deg = np.asarray(no_flip_rpy_deg, dtype=np.float64).copy()
            rpy_deg[2] = ((rpy_deg[2] + offset_deg + 180.0) % 360.0) - 180.0
            candidates.append((f"align_normal_no_flip_down_y_{offset_deg:+.1f}deg", rpy_deg))
        return candidates

    tolerance_deg = max(0.0, float(args.tool_y_tolerance_deg))
    offsets = [0.0, tolerance_deg * 0.5, -tolerance_deg * 0.5, tolerance_deg, -tolerance_deg]
    unique_offsets: list[float] = []
    for offset_deg in offsets:
        if not any(abs(offset_deg - existing) < 1e-6 for existing in unique_offsets):
            unique_offsets.append(offset_deg)

    current_rotation = rpy_xyz_to_matrix(current_pose.rpy_rad_xyz)
    max_rotation_deg = max(0.0, float(args.max_grasp_rotation_deg))
    for offset_deg in unique_offsets:
        rpy_deg = np.asarray(primary_rpy_deg, dtype=np.float64).copy()
        rpy_deg[2] = ((rpy_deg[2] + offset_deg + 180.0) % 360.0) - 180.0
        if rotation_distance_deg(current_rotation, rpy_xyz_to_matrix(np.radians(rpy_deg))) <= max_rotation_deg:
            candidates.append((f"{primary_rpy_mode}_selected_y_{offset_deg:+.1f}deg", rpy_deg))

    if args.tool_y_try_opposite:
        for offset_deg in unique_offsets:
            rpy_deg = np.asarray(primary_rpy_deg, dtype=np.float64).copy()
            rpy_deg[2] = ((rpy_deg[2] + 180.0 + offset_deg + 180.0) % 360.0) - 180.0
            if rotation_distance_deg(current_rotation, rpy_xyz_to_matrix(np.radians(rpy_deg))) <= max_rotation_deg:
                candidates.append((f"{primary_rpy_mode}_opposite_y_{offset_deg:+.1f}deg", rpy_deg))

    no_flip_offsets = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0]
    for offset_deg in no_flip_offsets:
        rpy_deg = np.asarray(no_flip_rpy_deg, dtype=np.float64).copy()
        rpy_deg[2] = ((rpy_deg[2] + offset_deg + 180.0) % 360.0) - 180.0
        if not any(
            rotation_distance_deg(
                rpy_xyz_to_matrix(np.radians(rpy_deg)),
                rpy_xyz_to_matrix(np.radians(existing_rpy_deg)),
            )
            < 1e-3
            for _existing_mode, existing_rpy_deg in candidates
        ):
            candidates.append((f"align_normal_no_flip_down_y_{offset_deg:+.1f}deg", rpy_deg))
    return candidates


def rotation_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cos_angle = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def unwrap_angle_deg(target_deg: float, reference_deg: float) -> float:
    """Return the target's equivalent angle nearest to the reference angle."""
    return float(reference_deg + ((target_deg - reference_deg + 180.0) % 360.0) - 180.0)


def unwrap_rpy_deg(target_rpy_deg: np.ndarray, reference_rpy_deg: np.ndarray) -> np.ndarray:
    """Keep Euler values continuous across the +/-180 degree boundary."""
    target = np.asarray(target_rpy_deg, dtype=np.float64)
    reference = np.asarray(reference_rpy_deg, dtype=np.float64)
    return np.asarray(
        [unwrap_angle_deg(float(target[index]), float(reference[index])) for index in range(3)],
        dtype=np.float64,
    )


def require_safe_rpy_step(
    previous_rpy_deg: np.ndarray,
    target_rpy_deg: np.ndarray,
    *,
    label: str,
    max_component_step_deg: float = 120.0,
) -> None:
    """Reject a genuinely large orientation step without Euler gimbal-lock false positives."""
    previous = np.asarray(previous_rpy_deg, dtype=np.float64)
    target = np.asarray(target_rpy_deg, dtype=np.float64)
    delta = np.abs(target - previous)
    rotation_step_deg = rotation_distance_deg(
        rpy_xyz_to_matrix(np.radians(previous)),
        rpy_xyz_to_matrix(np.radians(target)),
    )
    if rotation_step_deg > max_component_step_deg:
        raise RuntimeError(
            f"Unsafe RPY step for {label}: previous={previous.round(2).tolist()} "
            f"target={target.round(2).tolist()} delta={delta.round(2).tolist()} deg "
            f"rotation_distance={rotation_step_deg:.2f} deg "
            f"limit={max_component_step_deg:.1f} deg"
        )


def compute_grasp_rpy_from_normal_and_y(
    normal_base: np.ndarray,
    short_axis_base: np.ndarray,
    current_rpy_rad: np.ndarray | None,
    tool_contact_axis: str = "plus-z",
    tool_face_reference_rpy_deg: np.ndarray | None = None,
) -> np.ndarray:
    normal = np.asarray(normal_base, dtype=np.float64)
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    z_target = normal if tool_contact_axis == "minus-z" else -normal

    y_target = np.asarray(short_axis_base, dtype=np.float64)
    y_target -= z_target * float(np.dot(y_target, z_target))
    y_length = float(np.linalg.norm(y_target))
    if y_length < 1e-9:
        return compute_grasp_rpy_from_normal(
            normal, current_rpy_rad, tool_contact_axis, tool_face_reference_rpy_deg
        )
    y_target /= y_length

    if current_rpy_rad is not None:
        current_rotation = rpy_xyz_to_matrix(current_rpy_rad)
        current_y = current_rotation[:, 1]
        current_y -= z_target * float(np.dot(current_y, z_target))
        current_y_length = float(np.linalg.norm(current_y))
        if current_y_length >= 1e-9:
            current_y /= current_y_length
            if float(np.dot(y_target, current_y)) < 0.0:
                y_target = -y_target

    x_target = np.cross(y_target, z_target)
    x_length = float(np.linalg.norm(x_target))
    if x_length < 1e-9:
        return compute_grasp_rpy_from_normal(
            normal, current_rpy_rad, tool_contact_axis, tool_face_reference_rpy_deg
        )
    x_target /= x_length
    y_target = np.cross(z_target, x_target)
    y_target /= max(1e-9, float(np.linalg.norm(y_target)))

    target_rotation = np.column_stack([x_target, y_target, z_target])
    target_rotation = target_rotation @ tool_face_orientation_offset(
        tool_contact_axis, tool_face_reference_rpy_deg
    )
    return np.degrees(matrix_to_rpy_xyz(target_rotation))


def compute_grasp_rpy_from_normal_and_fixed_x(
    normal_base: np.ndarray,
    fixed_x_yaw_deg: float,
    tool_contact_axis: str = "plus-z",
) -> np.ndarray:
    """Align the suction contact axis while keeping a fixed base-frame X heading.

    Package OBB/Y direction and the current tool Y direction are deliberately
    ignored. For a horizontal top surface this is exactly
    RPY=[0, 0, fixed_x_yaw_deg] when tool -Z is the contact direction.
    """
    normal = np.asarray(normal_base, dtype=np.float64)
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    z_target = normal if tool_contact_axis == "minus-z" else -normal

    yaw_rad = np.radians(float(fixed_x_yaw_deg))
    x_reference = np.asarray([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float64)
    x_target = x_reference - z_target * float(np.dot(x_reference, z_target))
    x_length = float(np.linalg.norm(x_target))
    if x_length < 1e-9:
        x_reference = np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=np.float64)
        x_target = x_reference - z_target * float(np.dot(x_reference, z_target))
        x_length = float(np.linalg.norm(x_target))
    x_target /= max(1e-9, x_length)
    y_target = np.cross(z_target, x_target)
    y_target /= max(1e-9, float(np.linalg.norm(y_target)))
    x_target = np.cross(y_target, z_target)
    x_target /= max(1e-9, float(np.linalg.norm(x_target)))

    target_rotation = np.column_stack([x_target, y_target, z_target])
    return np.degrees(matrix_to_rpy_xyz(target_rotation))


def compute_grasp_rpy_from_normal(
    normal_base: np.ndarray,
    current_rpy_rad: np.ndarray,
    tool_contact_axis: str = "plus-z",
    tool_face_reference_rpy_deg: np.ndarray | None = None,
) -> np.ndarray:
    current_rotation = rpy_xyz_to_matrix(current_rpy_rad)

    normal = normal_base / np.linalg.norm(normal_base)
    contact_target = -normal
    contact_local = tool_contact_vector_local(
        tool_contact_axis, tool_face_reference_rpy_deg
    )
    contact_current = current_rotation @ contact_local
    contact_current = contact_current / np.linalg.norm(contact_current)

    cos_angle = float(np.clip(np.dot(contact_current, contact_target), -1.0, 1.0))
    if abs(cos_angle - 1.0) < 1e-9:
        new_rotation = current_rotation
    elif abs(cos_angle + 1.0) < 1e-9:
        axis = np.cross(contact_current, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(contact_current, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        k = skew(axis)
        delta = np.eye(3) + 2.0 * (k @ k)
        new_rotation = delta @ current_rotation
    else:
        axis = np.cross(contact_current, contact_target)
        axis = axis / np.linalg.norm(axis)
        sin_angle = np.linalg.norm(np.cross(contact_current, contact_target))
        k = skew(axis)
        delta = np.eye(3) + sin_angle * k + (1.0 - cos_angle) * (k @ k)
        new_rotation = delta @ current_rotation

    return np.degrees(matrix_to_rpy_xyz(new_rotation))


def tool_face_orientation_offset(
    tool_contact_axis: str,
    reference_rpy_deg: np.ndarray | None,
) -> np.ndarray:
    if reference_rpy_deg is None:
        return np.eye(3, dtype=np.float64)
    reference = np.asarray(reference_rpy_deg, dtype=np.float64)
    reference_rotation = rpy_xyz_to_matrix(np.radians(reference))
    reference_yaw = float(reference[2])
    canonical_rpy_deg = np.asarray(
        [0.0, 0.0, reference_yaw]
        if tool_contact_axis == "minus-z"
        else [180.0, 0.0, reference_yaw],
        dtype=np.float64,
    )
    canonical_rotation = rpy_xyz_to_matrix(np.radians(canonical_rpy_deg))
    return canonical_rotation.T @ reference_rotation


def tool_contact_vector_local(
    tool_contact_axis: str,
    reference_rpy_deg: np.ndarray | None,
) -> np.ndarray:
    canonical_contact = np.asarray(
        [0.0, 0.0, -1.0] if tool_contact_axis == "minus-z" else [0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    offset = tool_face_orientation_offset(tool_contact_axis, reference_rpy_deg)
    vector = offset.T @ canonical_contact
    return vector / np.linalg.norm(vector)


def skew(vec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ],
        dtype=np.float64,
    )


def build_region_seed_regions(
    depth_mm: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    component_labels: list[int],
    min_seed_area_px: int,
    reject_stats: dict[str, int],
    camera_matrix: np.ndarray,
    height_map: np.ndarray,
    split_peak_threshold_ratio: float,
    split_peak_min_distance_px: float,
) -> list[SeedRegion]:
    seed_regions: list[SeedRegion] = []
    for label in component_labels:
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < min_seed_area_px:
            reject_stats["cluster_size"] += 1
            continue

        component_mask = labels == label
        ys, xs = np.nonzero(component_mask)
        component_pixels = np.column_stack([xs, ys]).astype(np.int32)
        component_depth = depth_mm[component_mask]
        component_heights = height_map[component_mask]
        component_points = pixel_to_camera_points(component_pixels, component_depth, camera_matrix)

        seed_regions.extend(
            split_seed_component(
                component_mask,
                component_pixels,
                component_points,
                component_heights,
                label,
                min_seed_area_px,
                split_peak_threshold_ratio,
                split_peak_min_distance_px,
            )
        )
    return seed_regions


def build_instance_regions(
    depth_mm: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    component_labels: list[int],
    camera_matrix: np.ndarray,
    height_map: np.ndarray,
) -> list[SeedRegion]:
    instance_regions: list[SeedRegion] = []
    for label in component_labels:
        component_mask = labels == label
        ys, xs = np.nonzero(component_mask)
        if len(xs) == 0:
            continue
        component_pixels = np.column_stack([xs, ys]).astype(np.int32)
        component_depth = depth_mm[component_mask]
        component_heights = height_map[component_mask]
        component_points = pixel_to_camera_points(component_pixels, component_depth, camera_matrix)
        instance_regions.append(
            SeedRegion(
                label=label,
                area_px=int(stats[label, cv2.CC_STAT_AREA]),
                center_pixel=component_pixels.mean(axis=0),
                top_height_mm=float(np.percentile(component_heights, 95)),
                bbox_xyxy=region_bbox_xyxy(component_pixels),
                pixels=component_pixels,
                points=component_points,
                heights=component_heights,
            )
        )
    return instance_regions


def analyze_support_region(
    spec: SupportRegionSpec,
    region_polygon: np.ndarray,
    overall_mask: np.ndarray,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    camera_point_to_base: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
) -> SupportRegionAnalysis:
    polygon_mask = roi_mask_from_polygon(depth_mm.shape, region_polygon)
    if polygon_mask is None:
        return SupportRegionAnalysis(spec.region_id, spec.name, None, [], ["polygon=missing"])

    support_mask = polygon_mask & overall_mask & valid_mask
    support_pixels = int(np.count_nonzero(support_mask))
    if support_pixels < 80:
        return SupportRegionAnalysis(spec.region_id, spec.name, None, [], [f"support_pixels={support_pixels}", "support=too_small"])

    ys, xs = np.nonzero(support_mask)
    support_pixels_xy = np.column_stack([xs, ys]).astype(np.int32)
    support_depth = depth_mm[support_mask]
    support_points = pixel_to_camera_points(support_pixels_xy, support_depth, camera_matrix)

    try:
        plane = fit_plane_ransac(support_points, args.plane_threshold_mm, args.plane_iterations)
    except Exception as exc:
        return SupportRegionAnalysis(spec.region_id, spec.name, None, [], [f"plane_fit_failed={exc}"])

    _, height_map = dense_height_map(
        depth_mm,
        camera_matrix,
        plane,
        args.min_depth_mm,
        args.max_depth_mm,
    )
    object_mask = clean_object_mask(
        support_mask & (height_map > args.object_height_mm),
        args.mask_close_px,
        args.mask_open_px,
    )
    object_pixels_count = int(np.count_nonzero(object_mask))
    if object_pixels_count == 0:
        return SupportRegionAnalysis(
            spec.region_id,
            spec.name,
            plane,
            [],
            [f"support_pixels={support_pixels}", "object_pixels=0"],
        )

    height_gradient = compute_height_gradient(height_map, valid_mask)
    top_seed_mask = object_mask & (height_gradient <= args.max_top_gradient_mm_per_px)
    top_seed_mask = clean_object_mask(top_seed_mask, args.top_mask_close_px, 0)
    top_seed_pixels_count = int(np.count_nonzero(top_seed_mask))
    if top_seed_pixels_count == 0:
        return SupportRegionAnalysis(
            spec.region_id,
            spec.name,
            plane,
            [],
            [
                f"support_pixels={support_pixels}",
                f"object_pixels={object_pixels_count}",
                "top_seed_pixels=0",
            ],
        )

    instance_labels_count, instance_labels, instance_stats, _ = cv2.connectedComponentsWithStats(
        foreground.foreground_mask.astype(np.uint8),
        connectivity=8,
    )
    component_labels = sorted(
        range(1, instance_labels_count),
        key=lambda label: int(instance_stats[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    reject_stats = {
        "cluster_size": 0,
        "top_points": 0,
        "flatness": 0,
        "workspace": 0,
        "normal": 0,
    }
    seed_regions = build_region_seed_regions(
        depth_mm,
        labels,
        stats,
        component_labels,
        args.min_seed_area_px,
        reject_stats,
        camera_matrix,
        height_map,
        args.split_peak_threshold_ratio,
        args.split_peak_min_distance_px,
    )
    merged_regions = merge_seed_regions(
        seed_regions,
        args.merge_distance_px,
        args.merge_height_mm,
        args.merge_gap_px,
    )

    strict_candidates: list[ClusterCandidate] = []
    loose_candidates: list[ClusterCandidate] = []
    for region in merged_regions:
        if region.area_px < args.min_component_area_px:
            reject_stats["cluster_size"] += 1
            continue
        if len(region.points) < args.min_cluster_points or len(region.points) > args.max_cluster_points:
            reject_stats["cluster_size"] += 1
            continue

        top_height_mm = float(np.percentile(region.heights, 95))
        top_mask = region.heights >= (top_height_mm - args.top_slice_mm)
        top_points = region.points[top_mask]
        top_pixels = region.pixels[top_mask]
        if len(top_points) < args.min_top_points:
            reject_stats["top_points"] += 1
            continue

        top_plane = fit_top_plane_ransac_svd(top_points, args)
        if top_plane is None:
            reject_stats["top_points"] += 1
            continue
        top_points = top_points[top_plane.inlier_indices]
        top_pixels = top_pixels[top_plane.inlier_indices]
        top_centroid, top_normal = top_plane.centroid, top_plane.normal
        if float(np.dot(top_normal, plane.normal)) < 0.0:
            top_normal = -top_normal

        residuals = np.abs((top_points - top_centroid) @ top_normal)
        flatness_mm = float(residuals.mean())
        if flatness_mm > args.max_flatness_mm:
            reject_stats["flatness"] += 1
            continue

        center_pixel = tuple(np.round(np.median(top_pixels, axis=0)).astype(int))
        hull = oriented_box_from_pixels(top_pixels)
        point_base_mm = camera_point_to_base_mm(camera_point_to_base, top_centroid)
        normal_base = normal_camera_to_base(camera_point_to_base, top_normal)

        candidate = ClusterCandidate(
            index=0,
            center_pixel=center_pixel,
            hull_pixels=hull,
            support_region_id=spec.region_id,
            support_region_name=spec.name,
            point_camera_mm=top_centroid,
            point_base_mm=point_base_mm,
            normal_camera=top_normal,
            normal_base=normal_base,
            height_mm=top_height_mm,
            flatness_mm=flatness_mm,
            point_count=len(top_points),
            detection_source=rect_region.source,
            point_cloud_camera_mm=top_points,
            collision_target_mask=np.asarray(rect_region.mask, dtype=bool),
        )
        loose_candidates.append(candidate)

        workspace_ok = args.ignore_workspace_filter or candidate_in_workspace(
            point_base_mm,
            normal_base,
            args,
        )
        if not workspace_ok:
            reject_stats["workspace"] += 1
            continue
        if normal_base[2] < args.min_upward_normal_z:
            reject_stats["normal"] += 1
            continue
        strict_candidates.append(candidate)

    candidates = strict_candidates if strict_candidates else loose_candidates
    notes = [
        f"{spec.region_id}:support_pixels={support_pixels}",
        f"{spec.region_id}:object_pixels={object_pixels_count} top_seed_pixels={top_seed_pixels_count}",
        (
            f"{spec.region_id}:components={len(component_labels)} "
            f"merged={len(merged_regions)} "
            f"rejected=cluster_size:{reject_stats['cluster_size']} "
            f"top_points:{reject_stats['top_points']} "
            f"flatness:{reject_stats['flatness']} "
            f"workspace:{reject_stats['workspace']} "
            f"normal:{reject_stats['normal']}"
        ),
    ]
    if not strict_candidates and loose_candidates:
        notes.append(f"{spec.region_id}:using_relaxed_candidates")
    return SupportRegionAnalysis(spec.region_id, spec.name, plane, candidates, notes)


def analyze_scene_yolo_only(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    depth_display: np.ndarray,
    camera_matrix: np.ndarray,
    camera_point_to_base: np.ndarray,
    args: argparse.Namespace,
    roi_config: RoiConfig,
    yolo_model,
) -> AnalysisResult:
    depth_mm, depth_display = resize_depth_to_color_if_needed(depth_mm, depth_display, color_bgr.shape[:2])
    rgb_regions, yolo_notes, debug_masks = find_yolo_rectangle_regions(
        color_bgr,
        roi_config,
        args,
        yolo_model,
        exclude_mask=None,
    )

    reject_stats = {
        "rgb_size": 0,
        "depth": 0,
        "top_points": 0,
        "flatness": 0,
        "workspace": 0,
        "normal": 0,
    }
    loose_candidates: list[ClusterCandidate] = []
    foreground_ratio_notes: list[str] = []
    base_z_camera = normal_base_to_camera(camera_point_to_base, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))

    erode_kernel = None
    if args.yolo_point_erode_px > 0:
        size = args.yolo_point_erode_px * 2 + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for rect_region in rgb_regions:
        rect_area_px = int(np.count_nonzero(rect_region.mask))
        if rect_area_px < args.min_component_area_px:
            reject_stats["rgb_size"] += 1
            continue

        point_mask = rect_region.point_mask if rect_region.point_mask is not None else rect_region.mask
        if erode_kernel is not None:
            eroded = cv2.erode(point_mask.astype(np.uint8) * 255, erode_kernel) > 0
            if int(np.count_nonzero(eroded)) >= args.min_rect_foreground_pixels:
                point_mask = eroded

        valid_depth_mask = (
            point_mask
            & (depth_mm > args.min_depth_mm)
            & (depth_mm < args.max_depth_mm)
            & np.isfinite(depth_mm)
        )
        foreground_pixels = int(np.count_nonzero(valid_depth_mask))
        foreground_ratio = foreground_pixels / max(1, rect_area_px)
        if len(foreground_ratio_notes) < 8:
            foreground_ratio_notes.append(f"{rect_region.source}:{foreground_pixels}px/{foreground_ratio:.2f}")
        if foreground_pixels < args.min_rect_foreground_pixels:
            reject_stats["depth"] += 1
            continue

        ys, xs = np.nonzero(valid_depth_mask)
        rect_pixels = np.column_stack([xs, ys]).astype(np.int32)
        rect_depth = depth_mm[valid_depth_mask]
        rect_points = pixel_to_camera_points(rect_pixels, rect_depth, camera_matrix)
        if len(rect_points) < args.min_top_points:
            reject_stats["top_points"] += 1
            continue

        top_plane = fit_top_plane_ransac_svd(rect_points, args)
        if top_plane is None:
            reject_stats["top_points"] += 1
            continue
        rect_points = rect_points[top_plane.inlier_indices]
        rect_pixels = rect_pixels[top_plane.inlier_indices]
        top_centroid, top_normal = top_plane.centroid, top_plane.normal
        top_normal_base = normal_camera_to_base(camera_point_to_base, top_normal)
        if top_normal_base[2] < 0.0:
            top_normal = -top_normal
            top_normal_base = -top_normal_base

        residuals = np.abs((rect_points - top_centroid) @ top_normal)
        flatness_mm = float(residuals.mean())
        if flatness_mm > max(args.max_flatness_mm * 3.0, 15.0):
            reject_stats["flatness"] += 1
            continue

        grasp_center_pixel_np = (
            rect_region.center_pixel
            if args.grasp_center_mode == "yolo-center"
            else np.median(rect_pixels, axis=0)
        )
        point_camera_mm = top_centroid
        if args.grasp_center_mode == "yolo-center":
            projected_center = ray_plane_intersection_from_pixel(
                grasp_center_pixel_np,
                camera_matrix,
                top_centroid,
                top_normal,
                args.min_depth_mm,
                args.max_depth_mm,
            )
            if projected_center is not None:
                point_camera_mm = projected_center

        normal_camera, normal_base, normal_mode_used = select_grasp_normal(
            top_normal,
            base_z_camera,
            camera_point_to_base,
            args,
        )
        point_base_mm = camera_point_to_base_mm(camera_point_to_base, point_camera_mm)
        short_axis_camera, short_axis_base = short_axis_base_from_rectangle(
            rect_region.hull_pixels,
            camera_matrix,
            point_camera_mm,
            top_normal,
            camera_point_to_base,
            normal_base,
            args.min_depth_mm,
            args.max_depth_mm,
        )
        workspace_ok = args.ignore_workspace_filter or candidate_in_workspace(point_base_mm, normal_base, args)
        normal_ok = normal_base[2] >= args.min_upward_normal_z

        filter_reasons: list[str] = []
        if not workspace_ok:
            reject_stats["workspace"] += 1
            filter_reasons.append("workspace")
        if not normal_ok:
            reject_stats["normal"] += 1
            filter_reasons.append("normal")
        motion_safe = not filter_reasons

        candidate = ClusterCandidate(
            index=0,
            center_pixel=tuple(np.round(grasp_center_pixel_np).astype(int)),
            hull_pixels=rect_region.hull_pixels,
            support_region_id="yolo",
            support_region_name="YOLO",
            point_camera_mm=point_camera_mm,
            point_base_mm=point_base_mm,
            normal_camera=normal_camera,
            normal_base=normal_base,
            height_mm=float(point_base_mm[2]),
            flatness_mm=flatness_mm,
            point_count=len(rect_points),
            rect_area_px=float(rect_area_px),
            foreground_ratio=float(foreground_ratio),
            detection_source=f"{rect_region.source}/{normal_mode_used}",
            class_id=rect_region.class_id,
            class_name=rect_region.class_name,
            confidence=rect_region.confidence,
            motion_safe=motion_safe,
            filter_note="ok" if motion_safe else ",".join(filter_reasons),
            point_cloud_camera_mm=rect_points,
            collision_target_mask=np.asarray(rect_region.mask, dtype=bool),
            short_axis_camera=short_axis_camera,
            short_axis_base=short_axis_base,
        )
        loose_candidates.append(candidate)

    candidates = prune_duplicate_candidates(
        loose_candidates,
        color_bgr.shape[:2],
        args.final_candidate_nms_overlap,
        args.final_candidate_nms_min_area_ratio,
        args.final_candidate_nms_max_center_distance,
    )
    candidates.sort(key=lambda item: (-item.point_base_mm[2], item.flatness_mm, -item.point_count))
    for idx, candidate in enumerate(candidates, start=1):
        candidate.index = idx
    candidates, coverage = rank_candidates_for_next_pick(
        candidates,
        color_bgr.shape[:2],
        max_occlusion_ratio=args.pick_max_occlusion_ratio,
        height_weight=args.pick_height_weight,
        occlusion_weight=args.pick_occlusion_weight,
        flatness_weight=args.pick_flatness_weight,
        confidence_weight=args.pick_confidence_weight,
        point_quality_weight=args.pick_point_quality_weight,
        nested_small_bonus=args.pick_nested_small_bonus,
    )
    for idx, candidate in enumerate(candidates, start=1):
        candidate.index = idx

    notes = [
        "mode=yolo_only+point_cloud_plane",
        f"grasp_center_mode={args.grasp_center_mode}",
        (
            f"tool_y_mode={args.tool_y_mode} tolerance={args.tool_y_tolerance_deg:.1f}deg "
            f"opposite={args.tool_y_try_opposite} max_rotation={args.max_grasp_rotation_deg:.1f}deg"
        ),
        f"normal_mode={args.normal_mode} snap_angle={args.normal_snap_angle_deg:.1f}",
        f"overall_roi={'manual' if roi_config.overall_polygon is not None else 'auto'}",
        "support_regions=ignored",
        "exclude_rois=ignored",
        f"detector_regions={len(rgb_regions)}",
        f"depth_validated={len(loose_candidates)} pruned={len(candidates)} motion_safe={sum(1 for item in candidates if item.motion_safe)}",
        (
            "workspace_mm="
            f"x[{args.workspace_x_min_mm:.0f},{args.workspace_x_max_mm:.0f}] "
            f"y[{args.workspace_y_min_mm:.0f},{args.workspace_y_max_mm:.0f}] "
            f"z[{args.workspace_z_min_mm:.0f},{args.workspace_z_max_mm:.0f}]"
        ),
        "height_metric=base_z",
        (
            "pick_score_weights="
            f"height:{args.pick_height_weight:.2f},occlusion:{args.pick_occlusion_weight:.2f},"
            f"flatness:{args.pick_flatness_weight:.2f},confidence:{args.pick_confidence_weight:.2f},"
            f"point_quality:{args.pick_point_quality_weight:.2f},nested_bonus:{args.pick_nested_small_bonus:.2f}"
        ),
        f"pick_max_occlusion_ratio={args.pick_max_occlusion_ratio:.2f}",
    ]
    covered_notes = []
    for candidate in candidates:
        covered, ratio, blocker_index = coverage[id(candidate)]
        if covered:
            covered_notes.append(
                f"#{candidate.index}:covered={ratio:.2f} by previous-height-rank #{blocker_index}"
            )
    if covered_notes:
        notes.append("occlusion=" + " ".join(covered_notes[:8]))
    if args.ignore_workspace_filter:
        notes.append("workspace_filter=ignored_for_dry_run")
    notes.extend(yolo_notes)
    if foreground_ratio_notes:
        notes.append("rect_depth=" + " ".join(foreground_ratio_notes))
    notes.append(
        "rejected="
        f"rgb_size:{reject_stats['rgb_size']} "
        f"depth:{reject_stats['depth']} "
        f"top_points:{reject_stats['top_points']} "
        f"flatness:{reject_stats['flatness']} "
        f"workspace:{reject_stats['workspace']} "
        f"normal:{reject_stats['normal']}"
    )
    if any(not candidate.motion_safe for candidate in candidates):
        notes.append("Unsafe candidates are shown for debugging but blocked for real robot motion.")
    if not candidates:
        notes.append("No YOLO candidate survived point-cloud fitting.")

    debug_masks["top_seed"] = debug_masks.get("package_color_raw", np.zeros(color_bgr.shape[:2], dtype=bool))
    debug_bgr = build_debug_montage(color_bgr, debug_masks, rgb_regions, candidates)
    attach_collision_scene(candidates, depth_mm, camera_matrix, camera_point_to_base, args)
    return AnalysisResult(color_bgr, depth_mm, depth_display, None, candidates, notes, debug_bgr)


def analyze_scene(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    depth_display: np.ndarray,
    camera_matrix: np.ndarray,
    camera_point_to_base: np.ndarray,
    args: argparse.Namespace,
    roi_config: RoiConfig,
    yolo_model=None,
) -> AnalysisResult:
    if args.detector == "yolo":
        if yolo_model is None:
            raise RuntimeError("YOLO detector was requested but no model was loaded.")
        return analyze_scene_yolo_only(
            color_bgr,
            depth_mm,
            depth_display,
            camera_matrix,
            camera_point_to_base,
            args,
            roi_config,
            yolo_model,
        )

    depth_mm, depth_display = resize_depth_to_color_if_needed(depth_mm, depth_display, color_bgr.shape[:2])
    exclude_mask = exclude_mask_from_config(color_bgr.shape[:2], roi_config)
    support_model = build_grasp_support_plane_model(
        depth_mm,
        camera_matrix,
        GraspRoiConfig(
            overall_polygon=roi_config.overall_polygon,
            support_polygons=roi_config.support_polygons,
        ),
        args.min_depth_mm,
        args.max_depth_mm,
        args.plane_threshold_mm,
        args.plane_iterations,
        args.tray_erode_px,
    )
    foreground = build_grasp_foreground_from_support_planes(
        depth_mm,
        camera_matrix,
        support_model,
        args.min_depth_mm,
        args.max_depth_mm,
        args.object_height_mm,
        args.mask_close_px,
        args.mask_open_px,
    )

    height_gradient = compute_foreground_height_gradient(
        foreground.nearest_height_mm_map,
        support_model.valid_mask,
    )
    top_seed_mask = foreground.foreground_mask & (height_gradient <= args.max_top_gradient_mm_per_px)
    if np.any(exclude_mask):
        foreground.foreground_mask[exclude_mask] = False
        top_seed_mask[exclude_mask] = False
        foreground.foreground_pixels = int(np.count_nonzero(foreground.foreground_mask))
    top_seed_mask = clean_object_mask(top_seed_mask, args.top_mask_close_px, 0)
    top_seed_pixels_count = int(np.count_nonzero(top_seed_mask))

    if args.detector == "yolo":
        if yolo_model is None:
            raise RuntimeError("YOLO detector was requested but no model was loaded.")
        rgb_regions, rgb_notes, debug_masks = find_yolo_rectangle_regions(
            color_bgr,
            roi_config,
            args,
            yolo_model,
            exclude_mask,
        )
    else:
        rgb_regions, rgb_notes, debug_masks = find_rgb_rectangle_regions(
            color_bgr,
            roi_config,
            args,
            foreground.foreground_mask,
            exclude_mask,
        )
    reject_stats = {
        "rgb_size": 0,
        "foreground": 0,
        "height": 0,
        "top_points": 0,
        "flatness": 0,
        "workspace": 0,
        "normal": 0,
    }

    strict_candidates: list[ClusterCandidate] = []
    loose_candidates: list[ClusterCandidate] = []
    dominant_region_counts: dict[str, int] = {}
    foreground_ratio_notes: list[str] = []

    for rect_region in rgb_regions:
        rect_area_px = int(np.count_nonzero(rect_region.mask))
        if rect_area_px < args.min_component_area_px:
            reject_stats["rgb_size"] += 1
            continue

        point_source_mask = rect_region.point_mask if rect_region.point_mask is not None else rect_region.mask
        direct_surface_source = rect_region.source.startswith("yolo") or rect_region.source in {
            "cardboard",
            "package_color",
            "color",
        }
        if direct_surface_source:
            rect_foreground_mask = (
                point_source_mask
                & support_model.valid_mask
                & np.isfinite(foreground.nearest_height_mm_map)
                & (depth_mm > args.min_depth_mm)
                & (depth_mm < args.max_depth_mm)
            )
        else:
            rect_foreground_mask = (
                rect_region.mask
                & foreground.foreground_mask
                & support_model.valid_mask
                & np.isfinite(foreground.nearest_height_mm_map)
            )
        foreground_pixels = int(np.count_nonzero(rect_foreground_mask))
        foreground_ratio = foreground_pixels / max(1, rect_area_px)
        if len(foreground_ratio_notes) < 8:
            foreground_ratio_notes.append(
                f"{rect_region.source}:{foreground_pixels}px/{foreground_ratio:.2f}"
            )
        if (
            foreground_pixels < args.min_rect_foreground_pixels
            or (foreground_ratio < args.min_rect_foreground_ratio and not direct_surface_source)
            or foreground_pixels > args.max_rect_foreground_pixels
        ):
            reject_stats["foreground"] += 1
            continue

        ys, xs = np.nonzero(rect_foreground_mask)
        rect_pixels = np.column_stack([xs, ys]).astype(np.int32)
        rect_depth = depth_mm[rect_foreground_mask]
        rect_points = pixel_to_camera_points(rect_pixels, rect_depth, camera_matrix)
        rect_heights = foreground.nearest_height_mm_map[rect_foreground_mask].astype(np.float64)

        if len(rect_points) < args.min_cluster_points or len(rect_points) > args.max_cluster_points:
            reject_stats["rgb_size"] += 1
            continue

        region_ids = foreground.nearest_region_id_map[rect_foreground_mask]
        valid_region_ids = region_ids[region_ids != ""]
        if len(valid_region_ids) == 0:
            dominant_region_id = "overall"
        else:
            unique_ids, counts = np.unique(valid_region_ids, return_counts=True)
            dominant_region_id = str(unique_ids[int(np.argmax(counts))])
        dominant_region_counts[dominant_region_id] = dominant_region_counts.get(dominant_region_id, 0) + 1

        support_plane = support_model.support_planes.get(dominant_region_id)
        if support_plane is None:
            support_plane = next(iter(support_model.support_planes.values()))
        support_spec = get_grasp_support_region_spec(dominant_region_id)
        support_name = support_spec.name if support_spec is not None else dominant_region_id.title()

        region_top_seed_mask = top_seed_mask[rect_foreground_mask]
        top_height_mm = float(np.percentile(rect_heights, 95))
        if top_height_mm < args.min_candidate_height_mm:
            reject_stats["height"] += 1
            continue
        top_mask = (rect_heights >= (top_height_mm - args.top_slice_mm)) & region_top_seed_mask
        if int(np.count_nonzero(top_mask)) < args.min_top_points:
            top_mask = rect_heights >= (top_height_mm - args.top_slice_mm)
        top_points = rect_points[top_mask]
        top_pixels = rect_pixels[top_mask]
        if len(top_points) < args.min_top_points:
            reject_stats["top_points"] += 1
            continue

        top_plane = fit_top_plane_ransac_svd(top_points, args)
        if top_plane is None:
            reject_stats["top_points"] += 1
            continue
        top_points = top_points[top_plane.inlier_indices]
        top_pixels = top_pixels[top_plane.inlier_indices]
        top_centroid, top_normal = top_plane.centroid, top_plane.normal
        if float(np.dot(top_normal, support_plane.normal)) < 0.0:
            top_normal = -top_normal

        residuals = np.abs((top_points - top_centroid) @ top_normal)
        flatness_mm = float(residuals.mean())
        if flatness_mm > max(args.max_flatness_mm * 3.0, 15.0):
            reject_stats["flatness"] += 1
            continue

        if args.grasp_center_mode == "yolo-center" and rect_region.source.startswith("yolo"):
            grasp_center_pixel_np = rect_region.center_pixel
        else:
            grasp_center_pixel_np = np.median(top_pixels, axis=0)
        center_pixel = tuple(np.round(grasp_center_pixel_np).astype(int))
        hull = rect_region.hull_pixels
        point_camera_mm = top_centroid
        if args.grasp_center_mode == "yolo-center" and rect_region.source.startswith("yolo"):
            projected_center = ray_plane_intersection_from_pixel(
                grasp_center_pixel_np,
                camera_matrix,
                top_centroid,
                top_normal,
                args.min_depth_mm,
                args.max_depth_mm,
            )
            if projected_center is not None:
                point_camera_mm = projected_center

        normal_camera, normal_base, normal_mode_used = select_grasp_normal(
            top_normal,
            support_plane.normal,
            camera_point_to_base,
            args,
        )
        point_base_mm = camera_point_to_base_mm(camera_point_to_base, point_camera_mm)
        workspace_ok = args.ignore_workspace_filter or candidate_in_workspace(
            point_base_mm,
            normal_base,
            args,
        )
        flatness_ok = flatness_mm <= args.max_flatness_mm
        normal_ok = normal_base[2] >= args.min_upward_normal_z
        filter_reasons: list[str] = []
        if not flatness_ok:
            reject_stats["flatness"] += 1
            filter_reasons.append("flatness")
        if not workspace_ok:
            reject_stats["workspace"] += 1
            filter_reasons.append("workspace")
        if not normal_ok:
            reject_stats["normal"] += 1
            filter_reasons.append("normal")
        motion_safe = not filter_reasons

        candidate = ClusterCandidate(
            index=0,
            center_pixel=center_pixel,
            hull_pixels=hull,
            support_region_id=dominant_region_id,
            support_region_name=support_name,
            point_camera_mm=point_camera_mm,
            point_base_mm=point_base_mm,
            normal_camera=normal_camera,
            normal_base=normal_base,
            height_mm=top_height_mm,
            flatness_mm=flatness_mm,
            point_count=len(top_points),
            rect_area_px=float(rect_area_px),
            foreground_ratio=float(foreground_ratio),
            detection_source=f"{rect_region.source}/{normal_mode_used}",
            class_id=rect_region.class_id,
            class_name=rect_region.class_name,
            confidence=rect_region.confidence,
            motion_safe=motion_safe,
            filter_note="ok" if motion_safe else ",".join(filter_reasons),
            point_cloud_camera_mm=top_points,
            collision_target_mask=np.asarray(rect_region.mask, dtype=bool),
        )
        loose_candidates.append(candidate)

        if motion_safe:
            strict_candidates.append(candidate)

    pruned_candidates = prune_duplicate_candidates(
        loose_candidates,
        color_bgr.shape[:2],
        args.final_candidate_nms_overlap,
        args.final_candidate_nms_min_area_ratio,
        args.final_candidate_nms_max_center_distance,
    )
    candidates = pruned_candidates
    fallback_plane = next(iter(support_model.support_planes.values()))
    notes = [
        f"mode={args.detector}+depth_validation",
        f"grasp_center_mode={args.grasp_center_mode}",
        f"normal_mode={args.normal_mode} snap_angle={args.normal_snap_angle_deg:.1f}",
        f"overall_roi={'manual' if roi_config.overall_polygon is not None else 'auto'}",
        f"exclude_rois={len(roi_config.exclude_polygons)}",
        f"support_regions={len(support_model.support_planes)}",
        f"foreground_pixels={foreground.foreground_pixels}",
        f"top_seed_pixels={top_seed_pixels_count}",
        f"detector_regions={len(rgb_regions)}",
        f"depth_validated={len(loose_candidates)} pruned={len(pruned_candidates)} motion_safe={sum(1 for item in pruned_candidates if item.motion_safe)}",
        (
            "workspace_mm="
            f"x[{args.workspace_x_min_mm:.0f},{args.workspace_x_max_mm:.0f}] "
            f"y[{args.workspace_y_min_mm:.0f},{args.workspace_y_max_mm:.0f}] "
            f"z[{args.workspace_z_min_mm:.0f},{args.workspace_z_max_mm:.0f}]"
        ),
    ]
    if args.ignore_workspace_filter:
        notes.append("workspace_filter=ignored_for_dry_run")
    if not roi_config.support_polygons:
        notes.append("support_roi_missing: press 1 to draw Floor ROI after equipment moved")
    notes.extend(rgb_notes)
    if foreground_ratio_notes:
        notes.append("rect_foreground=" + " ".join(foreground_ratio_notes))
    notes.extend(support_model.notes[:6])
    notes.append(
        "rejected="
        f"rgb_size:{reject_stats['rgb_size']} "
        f"foreground:{reject_stats['foreground']} "
        f"height:{reject_stats['height']} "
        f"top_points:{reject_stats['top_points']} "
        f"flatness:{reject_stats['flatness']} "
        f"workspace:{reject_stats['workspace']} "
        f"normal:{reject_stats['normal']}"
    )
    if dominant_region_counts:
        notes.append(
            "dominant_regions="
            + " ".join(f"{region_id}:{count}" for region_id, count in sorted(dominant_region_counts.items()))
        )
    if any(not candidate.motion_safe for candidate in candidates):
        notes.append("Unsafe candidates are shown for debugging but blocked for real robot motion.")
    if not candidates:
        notes.append("No detector candidate survived depth validation.")

    candidates.sort(key=lambda item: (-item.point_base_mm[2], -item.height_mm, item.flatness_mm, -item.point_count))
    for idx, candidate in enumerate(candidates, start=1):
        candidate.index = idx

    debug_masks["top_seed"] = top_seed_mask
    debug_bgr = build_debug_montage(color_bgr, debug_masks, rgb_regions, candidates)
    attach_collision_scene(candidates, depth_mm, camera_matrix, camera_point_to_base, args)
    return AnalysisResult(color_bgr, depth_mm, depth_display, fallback_plane, candidates, notes, debug_bgr)


def attach_collision_scene(candidates, depth_mm, camera_matrix, transform, args):
    if not getattr(args, "cup_volume_check", False):
        return
    # Retain all valid visible depth pixels, including undetected obstacles and
    # the selected parcel. Detection ROI/foreground masks must not erase hazards.
    pixels, points = sample_point_cloud(depth_mm, camera_matrix, 1, args.min_depth_mm, args.max_depth_mm)
    cloud = points @ transform[:3, :3].T + transform[:3, 3] * 1000.0
    cloud.setflags(write=False)
    for candidate in candidates:
        candidate.collision_scene_base_mm = cloud
        target_mask = getattr(candidate, "collision_target_mask", None)
        if target_mask is not None and target_mask.shape == depth_mm.shape:
            belongs_to_target = target_mask[pixels[:, 1], pixels[:, 0]]
            obstacle_cloud = cloud[~belongs_to_target]
            obstacle_cloud.setflags(write=False)
            candidate.collision_obstacle_scene_base_mm = obstacle_cloud
        else:
            # Older/non-detector candidates do not carry pixel ownership. The
            # contact-surface fallback below remains available for them.
            candidate.collision_obstacle_scene_base_mm = None
        surface_camera = getattr(candidate, "point_cloud_camera_mm", None)
        if surface_camera is None or not len(surface_camera):
            candidate.collision_target_surface_base_mm = None
            candidate.collision_target_surface_tree = None
        else:
            surface = np.asarray(surface_camera) @ transform[:3, :3].T + transform[:3, 3] * 1000.0
            surface.setflags(write=False)
            candidate.collision_target_surface_base_mm = surface
            candidate.collision_target_surface_tree = cKDTree(surface)


def pickup_cup_volume_clear(candidate, tcp_xyz_mm, tcp_rotation, args, selected_cup_name=None):
    if not getattr(args, "cup_volume_check", False):
        return True
    cloud = candidate.collision_scene_base_mm
    if cloud is None or not len(cloud):
        print("Rejecting pickup: cup volume check has no scene point cloud (unknown).")
        return False
    axis = args.tool_contact_axis
    face_offset = tool_face_orientation_offset(axis, args.tool_face_reference_rpy_deg)
    for cup in suction_cup_specs(args):
        center = tcp_xyz_mm + tcp_rotation @ cup.offset_tool_mm
        # Apply the same physical face-frame convention as grasp orientation.
        rotation = tcp_rotation @ cup.rotation_tool_from_cup @ face_offset.T
        tolerance = 2.0
        if cup.name == selected_cup_name:
            # The existing pickup can intentionally compress the selected cup
            # below its detected contact plane. Never exempt the whole parcel.
            compression = max(0.0, float(args.pickup_down_mm) - float(args.standoff_mm))
            tolerance += compression
        if tolerance >= 43.0:
            print("Rejecting pickup: contact allowance would consume the entire 43mm cup body.")
            return False
        check_cloud = cloud
        if cup.name == selected_cup_name:
            obstacle_cloud = getattr(candidate, "collision_obstacle_scene_base_mm", None)
            if obstacle_cloud is not None:
                # At the contact endpoint the selected cup intentionally
                # overlaps its target parcel. Exclude that detector instance,
                # not merely the sparse RANSAC top-plane inliers. Other cups
                # still use the complete cloud and therefore cannot descend
                # into the same parcel unnoticed.
                check_cloud = obstacle_cloud
            else:
                target_surface = getattr(candidate, "collision_target_surface_base_mm", None)
                target_tree = getattr(candidate, "collision_target_surface_tree", None)
                match_mm = float(getattr(args, "cup_contact_surface_match_mm", 3.0))
                if target_surface is None or not len(target_surface) or target_tree is None:
                    print("Rejecting pickup: selected cup has no target-instance cloud for contact exclusion.")
                    return False
                distance, _ = target_tree.query(cloud, k=1)
                check_cloud = cloud[distance > match_mm]
        result = check_cup_volume(check_cloud, center, rotation,
                                  body_z_sign=1 if axis == "minus-z" else -1,
                                  contact_tolerance_mm=tolerance)
        minimum_hits = int(getattr(args, "cup_volume_min_points", 5))
        if len(result.hit_indices) >= minimum_hits:
            print(f"Rejecting pickup: cup={cup.name} observed body points={len(result.hit_indices)} "
                  f"max depth from contact={result.max_depth_mm:.2f}mm; "
                  f"threshold={minimum_hits} contact tolerance={tolerance:g}mm.")
            return False
    return True


def draw_analysis_overlay(
    analysis: AnalysisResult,
    selected_index: int | None,
    roi_config: RoiConfig,
    roi_draft_points: list[tuple[int, int]],
    roi_edit_target: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    color_canvas = analysis.color_bgr.copy()
    depth_canvas = cv2.applyColorMap(analysis.depth_display, cv2.COLORMAP_JET)

    status_lines = [
        "Analysis mode (auto-refresh while idle)",
        "Left click: move selected safe target",
        "Enter: run batch, Space: stop, r/d: refresh, b: exclude, m/1/2/3/4: ROI, 5/6: left/right suction ROI",
        f"candidates={len(analysis.candidates)}",
        "filters: detector regions + depth validation; red boxes are motion-blocked",
    ]
    status_lines.extend(analysis.notes[:5])
    draw_lines(color_canvas, status_lines)
    draw_lines(depth_canvas, status_lines)
    draw_roi_overlay(color_canvas, roi_config, roi_draft_points, roi_edit_target)
    draw_roi_overlay(depth_canvas, roi_config, roi_draft_points, roi_edit_target)

    for candidate in analysis.candidates:
        if not candidate.motion_safe:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255) if candidate.index == selected_index else (0, 255, 0)
        cv2.polylines(color_canvas, [candidate.hull_pixels], True, color, 2)
        cv2.polylines(depth_canvas, [candidate.hull_pixels], True, color, 2)

        cx, cy = candidate.center_pixel
        draw_cross(color_canvas, (cx, cy), color)
        draw_cross(depth_canvas, (cx, cy), color)

        type_label = short_class_label(candidate.class_name)
        label = (
            f"{candidate.index}:{type_label} "
            f"{candidate.detection_source[:2]} H={candidate.height_mm:.0f} fg={candidate.foreground_ratio:.2f}"
        )
        if not candidate.motion_safe:
            label += f" !{candidate.filter_note}"
        cv2.putText(
            color_canvas,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_canvas,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return color_canvas, depth_canvas


def draw_live_overlay(
    color_bgr: np.ndarray,
    depth_display: np.ndarray,
    roi_config: RoiConfig,
    roi_draft_points: list[tuple[int, int]],
    roi_edit_target: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    color_canvas = color_bgr.copy()
    depth_canvas = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    if roi_edit_target is not None:
        if roi_edit_target == "overall":
            title = "Overall ROI"
        elif roi_edit_target == "exclude":
            title = "Exclude ROI"
        elif roi_edit_target in {"suction_left", "suction_right"}:
            title = f"{roi_edit_target.removeprefix('suction_').title()} suction ROI"
        else:
            spec = get_support_region_spec(roi_edit_target)
            title = f"{spec.name} ROI" if spec is not None else roi_edit_target
        lines = [
            f"{title} edit mode",
            "Left click: add ROI point, right click: undo",
            "c/Enter: save ROI, esc: cancel, x: clear selected ROI, q: quit",
        ]
    else:
        lines = [
            "Live preview",
            "idle; d/r: refresh, b: exclude, m: overall, 1/2/3/4: support, 5/6: left/right suction ROI",
            f"overall_roi={'on' if roi_config.overall_polygon is not None else 'off'}",
            f"exclude_rois={len(roi_config.exclude_polygons)}",
            f"suction_rois={','.join(sorted(roi_config.suction_zone_polygons)) or 'off'}",
        ]
    draw_lines(color_canvas, lines)
    draw_lines(depth_canvas, lines)
    draw_roi_overlay(color_canvas, roi_config, roi_draft_points, roi_edit_target)
    draw_roi_overlay(depth_canvas, roi_config, roi_draft_points, roi_edit_target)
    return color_canvas, depth_canvas


def draw_roi_overlay(
    image: np.ndarray,
    roi_config: RoiConfig,
    roi_draft_points: list[tuple[int, int]],
    roi_edit_target: str | None,
) -> None:
    if roi_config.overall_polygon is not None and len(roi_config.overall_polygon) >= 3:
        cv2.polylines(image, [roi_config.overall_polygon.astype(np.int32)], True, (255, 200, 0), 2)
        cv2.putText(
            image,
            "ROI",
            tuple(roi_config.overall_polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

    for index, polygon in enumerate(roi_config.exclude_polygons, start=1):
        if polygon is None or len(polygon) < 3:
            continue
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon.astype(np.int32)], (0, 0, 255))
        cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0, image)
        cv2.polylines(image, [polygon.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.putText(
            image,
            f"EX{index}",
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for spec in SUPPORT_REGION_SPECS:
        polygon = roi_config.support_polygons.get(spec.region_id)
        if polygon is None or len(polygon) < 3:
            continue
        cv2.polylines(image, [polygon.astype(np.int32)], True, spec.color_bgr, 2)
        cv2.putText(
            image,
            spec.key_hint,
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            spec.color_bgr,
            2,
            cv2.LINE_AA,
        )

    for zone_name, color in (("left", (255, 0, 255)), ("right", (0, 165, 255))):
        polygon = roi_config.suction_zone_polygons.get(zone_name)
        if polygon is None or len(polygon) < 3:
            continue
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon.astype(np.int32)], color)
        cv2.addWeighted(overlay, 0.12, image, 0.88, 0.0, image)
        cv2.polylines(image, [polygon.astype(np.int32)], True, color, 2)
        cv2.putText(
            image,
            f"SUCTION {zone_name.upper()}",
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if not roi_draft_points:
        return

    draft = np.asarray(roi_draft_points, dtype=np.int32)
    for point in draft:
        cv2.circle(image, tuple(point), 4, (0, 100, 255), -1)

    if len(draft) >= 2:
        cv2.polylines(image, [draft], False, (0, 100, 255), 2)

    if roi_edit_target is not None and len(draft) >= 3:
        title = "ROI" if roi_edit_target == "overall" else roi_edit_target
        cv2.putText(
            image,
            f"{title} points: {len(draft)}",
            (20, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )


def draw_lines(image: np.ndarray, lines: Iterable[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


def draw_cross(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.drawMarker(
        image,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )


def normalize_to_panel(values: np.ndarray, panel_size: int, margin: int) -> np.ndarray:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, panel_size // 2, dtype=np.int32)
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    if abs(vmax - vmin) < 1e-9:
        return np.full(values.shape, panel_size // 2, dtype=np.int32)
    scaled = margin + (values - vmin) * (panel_size - 2 * margin) / (vmax - vmin)
    return np.clip(np.round(scaled), 0, panel_size - 1).astype(np.int32)


def make_point_projection_panel(
    points_mm: np.ndarray,
    center_mm: np.ndarray,
    axes: tuple[int, int],
    title: str,
    size: tuple[int, int] = (420, 320),
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    if len(points_mm) == 0:
        cv2.putText(panel, f"{title}: no points", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
        return panel

    x_axis, y_axis = axes
    xs = normalize_to_panel(np.concatenate([points_mm[:, x_axis], [center_mm[x_axis]]]), width, 32)
    ys = normalize_to_panel(np.concatenate([points_mm[:, y_axis], [center_mm[y_axis]]]), height, 32)
    point_xs = xs[:-1]
    point_ys = height - 1 - ys[:-1]
    center_x = int(xs[-1])
    center_y = int(height - 1 - ys[-1])

    depth_values = points_mm[:, 2]
    colors = cv2.applyColorMap(
        cv2.normalize(depth_values, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    for px, py, color in zip(point_xs, point_ys, colors):
        cv2.circle(panel, (int(px), int(py)), 1, tuple(int(v) for v in color), -1, cv2.LINE_AA)

    cv2.drawMarker(panel, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(panel, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    axis_names = ("X", "Y", "Z")
    cv2.putText(
        panel,
        f"{axis_names[x_axis]} vs {axis_names[y_axis]} camera mm",
        (18, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return panel


def write_ascii_ply(path: Path, points_mm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points_mm)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points_mm:
            handle.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")


def save_candidate_point_cloud_debug(
    candidate: ClusterCandidate,
    args: argparse.Namespace,
) -> np.ndarray | None:
    points = candidate.point_cloud_camera_mm
    if points is None or len(points) == 0:
        print(f"Candidate #{candidate.index} has no point-cloud debug data.")
        return None

    output_dir = Path(args.point_cloud_debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"candidate_{candidate.index}_{tag}"
    png_path = output_dir / f"{stem}.png"
    ply_path = output_dir / f"{stem}.ply"

    center = candidate.point_camera_mm
    panels = [
        make_point_projection_panel(points, center, (0, 1), "Top surface point cloud: X-Y"),
        make_point_projection_panel(points, center, (0, 2), "Top surface point cloud: X-Z"),
        make_point_projection_panel(points, center, (1, 2), "Top surface point cloud: Y-Z"),
    ]
    info = np.full((320, 420, 3), 245, dtype=np.uint8)
    lines = [
        f"candidate #{candidate.index}",
        f"type: {candidate.class_name} cls={candidate.class_id} conf={candidate.confidence:.2f}",
        f"source: {candidate.detection_source}",
        f"points: {len(points)}",
        f"flatness: {candidate.flatness_mm:.2f} mm",
        f"camera center mm: {candidate.point_camera_mm.round(1).tolist()}",
        f"base center mm: {candidate.point_base_mm.round(1).tolist()}",
        f"camera normal: {candidate.normal_camera.round(4).tolist()}",
        f"base normal: {candidate.normal_base.round(4).tolist()}",
    ]
    y = 30
    for line in lines:
        cv2.putText(info, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y += 34

    montage = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], info])])
    cv2.imwrite(str(png_path), montage)
    write_ascii_ply(ply_path, points)
    print(f"Point cloud debug saved: {png_path}")
    print(f"Point cloud PLY saved: {ply_path}")
    print(
        f"Point cloud result #{candidate.index}: "
        f"type={candidate.class_name} "
        f"cls={candidate.class_id} "
        f"conf={candidate.confidence:.2f} "
        f"center_camera(mm)={candidate.point_camera_mm.round(1).tolist()} "
        f"center_base(mm)={candidate.point_base_mm.round(1).tolist()} "
        f"normal_camera={candidate.normal_camera.round(4).tolist()} "
        f"normal_base={candidate.normal_base.round(4).tolist()} "
        f"flatness={candidate.flatness_mm:.2f} mm"
    )
    return montage


def pick_candidate(
    x: int,
    y: int,
    candidates: list[ClusterCandidate],
) -> ClusterCandidate | None:
    point = (float(x), float(y))
    for candidate in candidates:
        if cv2.pointPolygonTest(candidate.hull_pixels.astype(np.float32), point, False) >= 0:
            return candidate

    if not candidates:
        return None

    centers = np.asarray([candidate.center_pixel for candidate in candidates], dtype=np.float64)
    distances = np.linalg.norm(centers - np.asarray([x, y], dtype=np.float64), axis=1)
    idx = int(np.argmin(distances))
    if len(candidates) == 1 and distances[idx] <= 180.0:
        return candidates[idx]
    if distances[idx] <= 90.0:
        return candidates[idx]
    return None


def nearest_candidate_distance(
    x: int,
    y: int,
    candidates: list[ClusterCandidate],
) -> tuple[ClusterCandidate | None, float]:
    if not candidates:
        return None, float("inf")
    centers = np.asarray([candidate.center_pixel for candidate in candidates], dtype=np.float64)
    distances = np.linalg.norm(centers - np.asarray([x, y], dtype=np.float64), axis=1)
    idx = int(np.argmin(distances))
    return candidates[idx], float(distances[idx])


def execute_candidate(
    candidate: ClusterCandidate,
    robot: XCoreRobotClient | None,
    motion_options: MotionOptions,
    dry_run: bool,
    args: argparse.Namespace,
) -> None:
    target_xyz_mm = target_point_for_candidate(
        candidate.point_base_mm,
        candidate.normal_base,
        args.standoff_mm,
        args.standoff_mode,
    )

    if robot is None or dry_run:
        print(
            f"[DRY-RUN{'/UNSAFE' if not candidate.motion_safe else ''}] target #{candidate.index}: "
            f"type={candidate.class_name} "
            f"XYZ(mm)={target_xyz_mm.round(1).tolist()} "
            f"surface(mm)={candidate.point_base_mm.round(1).tolist()} "
            f"normal={candidate.normal_base.round(4).tolist()} "
            f"standoff_mode={args.standoff_mode} "
            f"rpy_mode={args.rpy_mode} "
            f"filter={candidate.filter_note}"
        )
        return

    if not candidate.motion_safe:
        print(f"Candidate #{candidate.index} is blocked for real robot motion: {candidate.filter_note}")
        return

    current_pose = robot.read_current_pose()
    current_xyz_mm = current_pose.translation_mm()
    target_rpy_deg, rpy_mode = target_rpy_for_candidate(candidate, current_pose, args)
    rpy_candidates = target_rpy_candidates_for_candidate(
        candidate,
        current_pose,
        args,
        target_rpy_deg,
        rpy_mode,
    )
    travel_mm = float(np.linalg.norm(target_xyz_mm - current_xyz_mm))
    print(
        f"Moving candidate #{candidate.index}: "
        f"current XYZ(mm)={current_xyz_mm.round(1).tolist()} "
        f"target XYZ(mm)={target_xyz_mm.round(1).tolist()} "
        f"travel={travel_mm:.1f} mm "
        f"standoff_mode={args.standoff_mode} rpy_mode={rpy_mode}"
    )
    final_pose = None
    last_error: Exception | None = None
    selected_rpy_deg = target_rpy_deg
    selected_rpy_mode = rpy_mode
    for candidate_rpy_mode, candidate_rpy_deg in rpy_candidates:
        selected_rpy_deg = candidate_rpy_deg
        selected_rpy_mode = candidate_rpy_mode
        try:
            print(
                f"Trying click-move orientation {candidate_rpy_mode}: "
                f"RPY(deg)={candidate_rpy_deg.round(2).tolist()}"
            )
            final_pose = move_pose_with_singularity_fallback(
                robot,
                float(target_xyz_mm[0]),
                float(target_xyz_mm[1]),
                float(target_xyz_mm[2]),
                float(candidate_rpy_deg[0]),
                float(candidate_rpy_deg[1]),
                float(candidate_rpy_deg[2]),
                replace(motion_options, use_current_conf_data=True),
                allow_clear_confdata_retry=True,
                allow_movej_singularity_retry=False,
            )
            break
        except Exception as exc:
            last_error = exc
            if not is_no_ik_solution_error(exc):
                break
            print(
                f"Click-move orientation {candidate_rpy_mode} has no IK solution; "
                "trying the next allowed orientation."
            )

    if final_pose is None:
        print(
            f"Robot move failed for candidate #{candidate.index}: {last_error}\n"
            f"Target XYZ(mm)={target_xyz_mm.round(1).tolist()} "
            f"Last RPY(deg)={selected_rpy_deg.round(2).tolist()}"
        )
        return

    print(
        f"Moved to candidate #{candidate.index}: "
        f"XYZ(mm)={target_xyz_mm.round(1).tolist()} "
        f"RPY(deg)={selected_rpy_deg.round(2).tolist()} "
        f"mode={selected_rpy_mode}"
    )
    final_xyz_mm = final_pose.translation_mm()
    final_error_mm = float(np.linalg.norm(final_xyz_mm - target_xyz_mm))
    print(
        "Final robot pose: "
        f"XYZ(mm)={final_xyz_mm.round(1).tolist()} "
        f"RPY(deg)={final_pose.rpy_deg_xyz().round(2).tolist()} "
        f"target_error={final_error_mm:.1f} mm"
    )
    if final_error_mm > 10.0:
        print(
            f"Warning: robot reported motion complete but final pose is {final_error_mm:.1f} mm from target."
        )


def move_waypoint(
    waypoint: RobotWaypoint,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    override_rpy_deg: np.ndarray | None = None,
) -> None:
    raw_rpy_deg = (
        np.asarray(override_rpy_deg, dtype=np.float64)
        if override_rpy_deg is not None
        else np.asarray([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg], dtype=np.float64)
    )
    # Fixed taught Euler angles are commonly reported in the canonical
    # [-180, 180] range.  Reusing those raw values after a failed suction
    # plan can turn an equivalent +91 degree return into a -269 degree MoveJ
    # command (for example +173.91 -> A* -94.661).  Make every fixed-waypoint
    # command continuous from the robot's actual pose, in both directions.
    current_rpy_deg = robot.read_current_pose().rpy_deg_xyz()
    continuous_rpy_deg = unwrap_rpy_deg(raw_rpy_deg, current_rpy_deg)
    require_safe_rpy_step(
        current_rpy_deg,
        continuous_rpy_deg,
        label=f"current pose -> waypoint {waypoint.name}",
    )
    print(
        f"Moving waypoint {waypoint.name}: "
        f"XYZ(mm)={[waypoint.x_mm, waypoint.y_mm, waypoint.z_mm]} "
        f"raw RPY(deg)={raw_rpy_deg.round(3).tolist()} "
        f"continuous RPY(deg)={continuous_rpy_deg.round(3).tolist()} "
        f"motion={motion_options.motion}"
    )
    move_pose_with_singularity_fallback(
        robot,
        waypoint.x_mm,
        waypoint.y_mm,
        waypoint.z_mm,
        float(continuous_rpy_deg[0]),
        float(continuous_rpy_deg[1]),
        float(continuous_rpy_deg[2]),
        motion_options,
        allow_clear_confdata_retry=not motion_options.use_current_conf_data,
        allow_movej_singularity_retry=not motion_options.use_current_conf_data,
    )


def pose_is_at_waypoint(
    pose,
    waypoint: RobotWaypoint,
    *,
    position_tolerance_mm: float = 5.0,
    orientation_tolerance_deg: float = 5.0,
) -> bool:
    target_xyz_mm = np.asarray(
        [waypoint.x_mm, waypoint.y_mm, waypoint.z_mm],
        dtype=np.float64,
    )
    target_rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    return bool(
        float(np.linalg.norm(pose.translation_mm() - target_xyz_mm))
        <= position_tolerance_mm
        and rotation_distance_deg(
            rpy_xyz_to_matrix(pose.rpy_rad_xyz),
            target_rotation,
        )
        <= orientation_tolerance_deg
    )


def waypoint_path_pose(waypoint: RobotWaypoint, zone_mm: float) -> tuple[float, float, float, float, float, float, float]:
    return (
        waypoint.x_mm,
        waypoint.y_mm,
        waypoint.z_mm,
        waypoint.rx_deg,
        waypoint.ry_deg,
        waypoint.rz_deg,
        zone_mm,
    )


def move_loaded_transfer_with_singularity_fallback(
    waypoint: RobotWaypoint,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    *,
    allow_current_conf_movej_fallback: bool = True,
) -> None:
    """Prefer loaded MoveL; on path singularity use MoveJ in the same configuration."""
    raw_rpy_deg = np.asarray(
        [waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg],
        dtype=np.float64,
    )
    current_rpy_deg = robot.read_current_pose().rpy_deg_xyz()
    continuous_rpy_deg = unwrap_rpy_deg(raw_rpy_deg, current_rpy_deg)
    require_safe_rpy_step(
        current_rpy_deg,
        continuous_rpy_deg,
        label=f"loaded current pose -> {waypoint.name}",
    )
    print(
        f"Moving loaded transfer {waypoint.name}: "
        f"XYZ(mm)={[round(waypoint.x_mm, 3), round(waypoint.y_mm, 3), round(waypoint.z_mm, 3)]} "
        f"raw RPY(deg)={raw_rpy_deg.round(3).tolist()} "
        f"continuous RPY(deg)={continuous_rpy_deg.round(3).tolist()} "
        f"motion={'movej-auto-conf' if motion_options.motion == 'movej' and not motion_options.use_current_conf_data else 'movel'}"
    )


    # The C-point barcode side view was verified as a MoveJ with controller
    # configuration selection enabled. Do not override that explicit request
    # with the ordinary loaded-transfer MoveL/current-conf policy.
    if motion_options.motion == "movej" and not motion_options.use_current_conf_data:
        print(
            f"Moving verified side-view transfer {waypoint.name} with MoveJ "
            "and automatic confData."
        )
        move_pose_with_singularity_fallback(
            robot,
            waypoint.x_mm,
            waypoint.y_mm,
            waypoint.z_mm,
            float(continuous_rpy_deg[0]),
            float(continuous_rpy_deg[1]),
            float(continuous_rpy_deg[2]),
            replace(motion_options, motion="movej", zone_mm=0.0),
            allow_clear_confdata_retry=True,
            allow_movej_singularity_retry=False,
        )
        return
    try:
        move_pose_with_singularity_fallback(
            robot,
            waypoint.x_mm,
            waypoint.y_mm,
            waypoint.z_mm,
            float(continuous_rpy_deg[0]),
            float(continuous_rpy_deg[1]),
            float(continuous_rpy_deg[2]),
            replace(motion_options, motion="movel", zone_mm=0.0, use_current_conf_data=True),
            # This wrapper owns the only permitted fallback: MoveJ with the
            # current configuration. Never let the generic helper clear
            # confData or choose a remote joint branch for a loaded package.
            allow_clear_confdata_retry=False,
            allow_movej_singularity_retry=False,
        )
    except Exception as exc:
        # A Cartesian path can cross a singularity even when the endpoint has
        # a valid solution. Only that specific path error may switch to MoveJ,
        # and the current configuration must remain locked.
        message = str(exc)
        if "50102" not in message and "奇异点" not in message:
            raise
        if not allow_current_conf_movej_fallback:
            raise RuntimeError(
                f"Strict loaded rotation to {waypoint.name} was rejected; "
                "MoveJ fallback is disabled for this rotation segment. "
                f"Controller error: {exc}"
            ) from exc
        print(
            f"Loaded MoveL to {waypoint.name} crosses a Cartesian path singularity; "
            "stopping that trajectory and retrying MoveJ with the current confData and "
            "the same continuous orientation."
        )
        robot.stop_motion()
        robot.move_to_pose_mm_deg(
            waypoint.x_mm,
            waypoint.y_mm,
            waypoint.z_mm,
            float(continuous_rpy_deg[0]),
            float(continuous_rpy_deg[1]),
            float(continuous_rpy_deg[2]),
            options=replace(
                motion_options,
                motion="movej",
                zone_mm=0.0,
                use_current_conf_data=True,
            ),
        )


def interpolated_orientation_steps_deg(
    start_rpy_deg: np.ndarray,
    target_rpy_deg: np.ndarray,
    max_step_deg: float = 30.0,
) -> list[np.ndarray]:
    """Interpolate the shortest physical rotation into bounded RPY targets."""
    start_rotation = rpy_xyz_to_matrix(np.radians(np.asarray(start_rpy_deg, dtype=np.float64)))
    target_rotation = rpy_xyz_to_matrix(np.radians(np.asarray(target_rpy_deg, dtype=np.float64)))
    relative_rotvec = Rotation.from_matrix(start_rotation.T @ target_rotation).as_rotvec()
    total_deg = float(np.degrees(np.linalg.norm(relative_rotvec)))
    step_count = max(1, int(np.ceil(total_deg / max_step_deg)))
    steps: list[np.ndarray] = []
    previous = np.asarray(start_rpy_deg, dtype=np.float64)
    for index in range(1, step_count + 1):
        fraction = index / step_count
        rotation = start_rotation @ Rotation.from_rotvec(relative_rotvec * fraction).as_matrix()
        raw_rpy = np.degrees(matrix_to_rpy_xyz(rotation))
        continuous_rpy = unwrap_rpy_deg(raw_rpy, previous)
        steps.append(continuous_rpy)
        previous = continuous_rpy
    return steps


def recover_empty_tool_from_d_to_b(
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
) -> None:
    """Recover a restarted empty robot from a D-area pose through the safe high point."""
    current_pose = robot.read_current_pose()
    current_xyz_mm = current_pose.translation_mm()
    d_xyz_mm = np.asarray([WAYPOINT_D.x_mm, WAYPOINT_D.y_mm, WAYPOINT_D.z_mm])
    if float(np.linalg.norm(current_xyz_mm - d_xyz_mm)) > 400.0:
        raise RuntimeError(
            "The current pose is not near D, so the automatic D recovery route is not applicable."
        )
    current_rpy_deg = current_pose.rpy_deg_xyz()
    print(
        "Restart recovery: lifting the empty tool from the D area to D-rotate-safe "
        "without changing its orientation."
    )
    move_pose_with_singularity_fallback(
        robot,
        WAYPOINT_D_ROTATE_SAFE.x_mm,
        WAYPOINT_D_ROTATE_SAFE.y_mm,
        WAYPOINT_D_ROTATE_SAFE.z_mm,
        *[float(value) for value in current_rpy_deg],
        replace(motion_options, motion="movel", zone_mm=0.0, use_current_conf_data=True),
        allow_clear_confdata_retry=False,
        allow_movej_singularity_retry=False,
    )
    target_rpy_deg = np.asarray(
        [WAYPOINT_B.rx_deg, WAYPOINT_B.ry_deg, WAYPOINT_B.rz_deg], dtype=np.float64
    )
    steps = interpolated_orientation_steps_deg(current_rpy_deg, target_rpy_deg)
    previous_rpy_deg = current_rpy_deg
    for index, step_rpy_deg in enumerate(steps, start=1):
        require_safe_rpy_step(
            previous_rpy_deg,
            step_rpy_deg,
            label=f"D restart recovery rotation {index}/{len(steps)}",
        )
        print(
            f"Restart recovery rotation {index}/{len(steps)} at D-rotate-safe: "
            f"RPY(deg)={step_rpy_deg.round(2).tolist()}."
        )
        move_pose_with_singularity_fallback(
            robot,
            WAYPOINT_D_ROTATE_SAFE.x_mm,
            WAYPOINT_D_ROTATE_SAFE.y_mm,
            WAYPOINT_D_ROTATE_SAFE.z_mm,
            *[float(value) for value in step_rpy_deg],
            replace(motion_options, motion="movel", zone_mm=0.0, use_current_conf_data=True),
            allow_clear_confdata_retry=False,
            allow_movej_singularity_retry=False,
        )
        previous_rpy_deg = step_rpy_deg
    move_waypoint(WAYPOINT_B, robot, motion_options)


def move_pose_with_singularity_fallback(
    robot: XCoreRobotClient,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    motion_options: MotionOptions,
    allow_clear_confdata_retry: bool = True,
    allow_movej_singularity_retry: bool = True,
):
    try:
        return robot.move_to_pose_mm_deg(
            x_mm,
            y_mm,
            z_mm,
            rx_deg,
            ry_deg,
            rz_deg,
            options=motion_options,
        )
    except Exception as exc:
        message = str(exc)
        if "-50021" in message or "50021" in message:
            if not motion_options.use_current_conf_data or not allow_clear_confdata_retry:
                raise
            fallback_options = replace(
                motion_options,
                use_current_conf_data=False,
            )
            print(
                f"Target rejected with current confData for "
                f"XYZ(mm)={[x_mm, y_mm, z_mm]}; "
                "retrying without confData."
            )
            robot.stop_motion()
            return robot.move_to_pose_mm_deg(
                x_mm,
                y_mm,
                z_mm,
                rx_deg,
                ry_deg,
                rz_deg,
                options=fallback_options,
            )

        if "50102" not in message and "奇异点" not in message:
            raise
        if not allow_movej_singularity_retry:
            raise

        fallback_options = replace(
            motion_options,
            motion="movej",
            use_current_conf_data=False,
        )
        print(
            f"MoveL reported a singularity for target "
            f"XYZ(mm)={[x_mm, y_mm, z_mm]}; "
            "stopping the rejected trajectory and retrying MoveJ without confData."
        )
        robot.stop_motion()
        return robot.move_to_pose_mm_deg(
            x_mm,
            y_mm,
            z_mm,
            rx_deg,
            ry_deg,
            rz_deg,
            options=fallback_options,
        )


def move_empty_approach_to_a(
    robot: XCoreRobotClient,
    approach_xyz_mm: np.ndarray,
    rpy_deg: np.ndarray,
    motion_options: MotionOptions,
) -> np.ndarray:
    # A* is a physically taught and verified empty/loaded transition point.
    # Use joint interpolation from the normalized empty-tool pose through A*
    # to dynamic A; reserve Cartesian MoveL for the short A -> pickup motion.
    current_pose = robot.read_current_pose()
    at_a_star = pose_is_at_waypoint(current_pose, WAYPOINT_A_STAR)
    if at_a_star:
        print("Already at verified transition A*; skipping the redundant A* command.")
    else:
        print(
            "Moving through verified transition A*: "
            f"XYZ(mm)={[WAYPOINT_A_STAR.x_mm, WAYPOINT_A_STAR.y_mm, WAYPOINT_A_STAR.z_mm]} "
            f"RPY(deg)={[WAYPOINT_A_STAR.rx_deg, WAYPOINT_A_STAR.ry_deg, WAYPOINT_A_STAR.rz_deg]} motion=movej"
        )
        move_waypoint(
            WAYPOINT_A_STAR,
            robot,
            replace(motion_options, motion="movej", zone_mm=0.0, use_current_conf_data=True),
        )

    # The same Cartesian orientation may be written as +175.38 or -184.62
    # degrees. Always command the representation nearest to the actual A*
    # orientation; otherwise MoveJ can wind the wrist through +270 degrees.
    a_star_pose = robot.read_current_pose()
    continuous_rpy_deg = unwrap_rpy_deg(rpy_deg, a_star_pose.rpy_deg_xyz())
    require_safe_rpy_step(
        a_star_pose.rpy_deg_xyz(),
        continuous_rpy_deg,
        label="empty A* -> dynamic A",
    )
    approach_pose = (
        float(approach_xyz_mm[0]),
        float(approach_xyz_mm[1]),
        float(approach_xyz_mm[2]),
        float(continuous_rpy_deg[0]),
        float(continuous_rpy_deg[1]),
        float(continuous_rpy_deg[2]),
    )
    print(
        "Moving from A* to dynamic A: "
        f"XYZ(mm)={np.asarray(approach_pose[:3]).round(2).tolist()} "
        f"raw RPY(deg)={np.asarray(rpy_deg).round(2).tolist()} "
        f"continuous RPY(deg)={continuous_rpy_deg.round(2).tolist()} motion=movej"
    )
    move_pose_with_singularity_fallback(
        robot,
        *approach_pose,
        replace(motion_options, motion="movej", zone_mm=0.0, use_current_conf_data=True),
        # The controller can report -50021 solely because the current A*
        # confData excludes another valid IK branch. This is still an empty-
        # tool move, so retry the same ranked plan without confData before
        # rejecting that cup. Loaded motion remains configuration-locked.
        allow_clear_confdata_retry=True,
        allow_movej_singularity_retry=False,
    )
    return continuous_rpy_deg


def is_robot_power_or_safety_state_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "ec': -17" in message
        or 'ec": -17' in message
        or "上下电状态" in message
        or "power state" in message.lower()
    )


def is_operator_software_stop_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "operator software stop request" in message
        or "batch interrupted by operator" in message
    )


def wait_with_operator_stop(duration_s: float, robot: XCoreRobotClient, motion_options: MotionOptions) -> None:
    deadline = time.time() + max(0.0, duration_s)
    while time.time() < deadline:
        if motion_options.stop_requested is not None and motion_options.stop_requested():
            robot.stop_motion()
            raise RuntimeError("Batch interrupted by operator software stop request.")
        time.sleep(min(0.05, max(0.0, deadline - time.time())))


def receive_barcode_from_tcp_client(
    bind_ip: str,
    port: int,
    timeout_s: float,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> str | None:
    """Accept one reader connection and return its first non-empty payload."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_ip, int(port)))
        server.listen(1)

        while True:
            if stop_requested is not None and stop_requested():
                raise RuntimeError("Batch interrupted by operator software stop request.")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            server.settimeout(min(0.1, remaining))
            try:
                connection, _peer = server.accept()
            except socket.timeout:
                continue

            with connection:
                while True:
                    if stop_requested is not None and stop_requested():
                        raise RuntimeError("Batch interrupted by operator software stop request.")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    connection.settimeout(min(0.1, remaining))
                    try:
                        payload = connection.recv(65536)
                    except socket.timeout:
                        continue
                    if not payload:
                        break
                    payload = payload.strip(b"\x00\r\n\t ")
                    if not payload:
                        continue
                    try:
                        return payload.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            return payload.decode("gb18030")
                        except UnicodeDecodeError:
                            return payload.hex(" ")


def read_top_front_barcode_in_background(
    args: argparse.Namespace,
    motion_options: MotionOptions,
) -> str | None:
    """Receive one placed-package result without blocking robot motion."""
    if args.disable_barcode_reader:
        return None
    print(
        "D-rotate-safe reached; background listener is receiving top/front barcode data at "
        f"{args.barcode_reader_bind_ip}:{args.barcode_reader_port} for "
        f"up to {args.barcode_reader_timeout_s:.1f}s while the robot continues through C/B."
    )
    try:
        barcode_text = receive_barcode_from_tcp_client(
            args.barcode_reader_bind_ip,
            args.barcode_reader_port,
            args.barcode_reader_timeout_s,
            stop_requested=motion_options.stop_requested,
        )
    except Exception as barcode_exc:
        if is_operator_software_stop_error(barcode_exc):
            print("Top/front barcode background listener canceled by operator stop.")
            return None
        print(
            "Top/front barcode reader communication failed; label presence is unknown. "
            f"Error: {barcode_exc}"
        )
        return None
    if barcode_text is None:
        print(
            "No barcode data received by the background listener; "
            "the package top and front visible faces have no waybill."
        )
    else:
        print(f"Top/front barcode received: {barcode_text}")
    return barcode_text


def start_top_front_barcode_listener(
    args: argparse.Namespace,
    motion_options: MotionOptions,
) -> threading.Thread | None:
    """Start the D-point reader independently from the robot-motion thread."""
    if args.disable_barcode_reader:
        print("Top/front barcode reader is disabled; skipping TCP read after D placement.")
        return None
    listener = threading.Thread(
        target=read_top_front_barcode_in_background,
        args=(args, motion_options),
        name="top-front-barcode-listener",
        daemon=True,
    )
    listener.start()
    return listener


def suction_cup_specs(args: argparse.Namespace) -> list[SuctionCupSpec]:
    cups = [
        SuctionCupSpec(
            name="primary",
            do_port=int(args.suction_do_port),
            offset_tool_mm=np.zeros(3, dtype=np.float64),
        )
    ]
    if not args.disable_secondary_suction:
        cups.append(
            SuctionCupSpec(
                name="secondary",
                do_port=int(args.secondary_suction_do_port),
                offset_tool_mm=np.asarray(
                    [0.0, float(args.secondary_suction_offset_y_mm), 0.0],
                    dtype=np.float64,
                ),
            )
        )
    if not args.disable_third_suction:
        # Ry(-90) maps virtual-cup -Z onto main-tool +X and keeps Y aligned.
        cups.append(
            SuctionCupSpec(
                name="third",
                do_port=int(args.third_suction_do_port),
                offset_tool_mm=np.asarray(
                    args.third_suction_offset_xyz_mm,
                    dtype=np.float64,
                ),
                rotation_tool_from_cup=rpy_xyz_to_matrix(
                    np.radians([0.0, -90.0, 0.0])
                ),
            )
        )
    if not args.disable_fourth_suction:
        cups.append(
            SuctionCupSpec(
                name="fourth",
                do_port=int(args.fourth_suction_do_port),
                offset_tool_mm=np.asarray(
                    args.fourth_suction_offset_xyz_mm,
                    dtype=np.float64,
                ),
                rotation_tool_from_cup=rpy_xyz_to_matrix(
                    np.radians([0.0, -90.0, 0.0])
                ),
            )
        )
    return cups


def selected_suction_cup_specs(args: argparse.Namespace) -> list[SuctionCupSpec]:
    """Return cups allowed for pickup planning; IO shutdown still uses all cups."""
    cups = suction_cup_specs(args)
    forced_numbers = getattr(args, "force_suction_cups", None)
    if not forced_numbers:
        return cups
    number_to_name = {"1": "primary", "2": "secondary", "3": "third", "4": "fourth"}
    allowed_names = {number_to_name[number] for number in forced_numbers}
    selected = [cup for cup in cups if cup.name in allowed_names]
    if not selected:
        raise ValueError(
            "--force-suction-cups excludes every enabled suction cup; "
            "remove the matching --disable-*-suction option."
        )
    return selected


def suction_zone_for_candidate(
    candidate: ClusterCandidate,
    roi_config: RoiConfig | None,
) -> str | None:
    """Return the exclusive suction zone containing the grasp center."""
    if roi_config is None:
        return None
    point = (float(candidate.center_pixel[0]), float(candidate.center_pixel[1]))
    for zone_name in ("left", "right"):
        polygon = roi_config.suction_zone_polygons.get(zone_name)
        if polygon is not None and len(polygon) >= 3:
            if cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0.0:
                return zone_name
    return None


def suction_cups_for_candidate(
    candidate: ClusterCandidate,
    args: argparse.Namespace,
    roi_config: RoiConfig | None,
) -> tuple[list[SuctionCupSpec], str | None]:
    cups = selected_suction_cup_specs(args)
    zone_name = suction_zone_for_candidate(candidate, roi_config)
    if zone_name is None:
        return cups, None
    number_to_name = {"1": "primary", "2": "secondary", "3": "third", "4": "fourth"}
    configured_numbers = getattr(args, f"{zone_name}_zone_suction_cups")
    allowed_names = {number_to_name[number] for number in configured_numbers}
    return [cup for cup in cups if cup.name in allowed_names], zone_name


def tcp_rotation_for_cup_rotation(
    cup_rotation_base: np.ndarray,
    cup: SuctionCupSpec,
) -> np.ndarray:
    """Convert a desired virtual-cup orientation into the active TCP orientation."""
    return (
        np.asarray(cup_rotation_base, dtype=np.float64)
        @ np.asarray(cup.rotation_tool_from_cup, dtype=np.float64).T
    )


def cup_rotation_at_tcp_rotation(
    tcp_rotation_base: np.ndarray,
    cup: SuctionCupSpec,
) -> np.ndarray:
    """Return the selected virtual cup frame orientation for an active TCP pose."""
    return (
        np.asarray(tcp_rotation_base, dtype=np.float64)
        @ np.asarray(cup.rotation_tool_from_cup, dtype=np.float64)
    )


def set_suction_output(
    robot: XCoreRobotClient,
    args: argparse.Namespace,
    state: bool,
    *,
    do_port: int | None = None,
) -> None:
    port = int(args.suction_do_port if do_port is None else do_port)
    if args.disable_suction_io:
        print(f"Suction IO disabled; skip setting DO{args.suction_do_board}_{port}={state}.")
        return
    do_state = not state if args.invert_suction_io else state
    robot.set_do(args.suction_do_board, port, do_state)
    print(
        f"Suction command {'ON' if state else 'OFF'} -> "
        f"DO{args.suction_do_board}_{port} {'ON' if do_state else 'OFF'}."
    )


def set_all_suction_outputs(robot: XCoreRobotClient, args: argparse.Namespace, state: bool) -> None:
    # Disabling a cup removes it from planning, but must never remove its valve
    # from the all-off operation. A disabled valve could otherwise retain ON
    # from an interrupted previous run.
    physical_ports = (
        int(args.suction_do_port),
        int(args.secondary_suction_do_port),
        int(args.third_suction_do_port),
        int(args.fourth_suction_do_port),
    )
    for port in dict.fromkeys(physical_ports):
        set_suction_output(robot, args, state, do_port=port)


def candidate_collision_radius_mm(candidate: ClusterCandidate) -> float:
    """Conservative top-footprint radius estimated from the fitted package point cloud."""
    points = candidate.point_cloud_camera_mm
    if points is None or len(points) < 3:
        return 0.0
    center = np.asarray(candidate.point_camera_mm, dtype=np.float64)
    distances = np.linalg.norm(np.asarray(points, dtype=np.float64) - center, axis=1)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return 0.0
    # Ignore isolated depth outliers while retaining nearly the full package footprint.
    return float(np.percentile(finite, 98.0))


def candidate_footprint_clearance_mm(
    candidate: ClusterCandidate,
    query_base_mm: np.ndarray,
    cup_collision_radius_mm: float,
) -> float:
    """Signed XY edge clearance from an unused cup to an oriented package footprint."""
    points = candidate.point_cloud_camera_mm
    short_camera = candidate.short_axis_camera
    short_base = candidate.short_axis_base
    if points is None or len(points) < 3 or short_camera is None or short_base is None:
        center_distance = float(
            np.linalg.norm(np.asarray(query_base_mm, dtype=np.float64)[:2] - candidate.point_base_mm[:2])
        )
        return center_distance - candidate_collision_radius_mm(candidate) - cup_collision_radius_mm

    normal_camera = np.asarray(candidate.normal_camera, dtype=np.float64)
    short_camera = np.asarray(short_camera, dtype=np.float64)
    long_camera = np.cross(normal_camera, short_camera)
    long_camera_norm = float(np.linalg.norm(long_camera))
    short_camera_norm = float(np.linalg.norm(short_camera))
    short_base_xy = np.asarray(short_base, dtype=np.float64)[:2]
    short_base_norm = float(np.linalg.norm(short_base_xy))
    if long_camera_norm < 1e-9 or short_camera_norm < 1e-9 or short_base_norm < 1e-9:
        center_distance = float(
            np.linalg.norm(np.asarray(query_base_mm, dtype=np.float64)[:2] - candidate.point_base_mm[:2])
        )
        return center_distance - candidate_collision_radius_mm(candidate) - cup_collision_radius_mm

    short_camera /= short_camera_norm
    long_camera /= long_camera_norm
    short_base_xy /= short_base_norm
    long_base_xy = np.asarray([-short_base_xy[1], short_base_xy[0]], dtype=np.float64)
    centered_points = np.asarray(points, dtype=np.float64) - np.asarray(candidate.point_camera_mm, dtype=np.float64)
    half_short_mm = float(np.percentile(np.abs(centered_points @ short_camera), 98.0))
    half_long_mm = float(np.percentile(np.abs(centered_points @ long_camera), 98.0))

    delta_xy = np.asarray(query_base_mm, dtype=np.float64)[:2] - candidate.point_base_mm[:2]
    outside_long = abs(float(delta_xy @ long_base_xy)) - half_long_mm
    outside_short = abs(float(delta_xy @ short_base_xy)) - half_short_mm
    if outside_long <= 0.0 and outside_short <= 0.0:
        # Negative means the unused cup center lies inside the detected footprint.
        edge_distance = max(outside_long, outside_short)
    else:
        edge_distance = float(np.hypot(max(0.0, outside_long), max(0.0, outside_short)))
    return edge_distance - cup_collision_radius_mm


def build_suction_approach_plans(
    candidate: ClusterCandidate,
    scene_candidates: list[ClusterCandidate],
    current_tcp_xyz_mm: np.ndarray,
    current_tcp_rpy_deg: np.ndarray,
    physical_approach_xyz_mm: np.ndarray,
    physical_pickup_xyz_mm: np.ndarray,
    rpy_candidates: list[tuple[str, np.ndarray]],
    args: argparse.Namespace,
    roi_config: RoiConfig | None = None,
) -> list[SuctionApproachPlan]:
    """Rank TCP targets for every cup while keeping the chosen cup on the package."""
    plans: list[SuctionApproachPlan] = []
    cups, suction_zone = suction_cups_for_candidate(candidate, args, roi_config)
    if suction_zone is not None:
        print(
            f"Candidate #{candidate.index} center pixel={candidate.center_pixel} is in the "
            f"{suction_zone} suction ROI; allowed cups={[cup.name for cup in cups]}."
        )
    if not cups:
        print(
            f"Candidate #{candidate.index} has no enabled cup allowed by the "
            f"{suction_zone} suction ROI."
        )
        return []
    other_packages = [
        item
        for item in scene_candidates
        # A motion-blocked detection is still a real physical obstacle for
        # the unused suction cup.
        if item.index != candidate.index
    ]
    preferred_clearance_mm = max(0.0, float(args.dual_suction_clearance_mm))
    minimum_clearance_mm = max(0.0, float(getattr(args, "unused_cup_min_clearance_mm", 20.0)))
    cup_collision_radius_mm = max(0.0, float(getattr(args, "suction_cup_collision_radius_mm", 35.0)))
    a_star_xyz_mm = np.asarray(
        [WAYPOINT_A_STAR.x_mm, WAYPOINT_A_STAR.y_mm, WAYPOINT_A_STAR.z_mm],
        dtype=np.float64,
    )
    current_rotation = rpy_xyz_to_matrix(
        np.radians(np.asarray(current_tcp_rpy_deg, dtype=np.float64))
    )

    for rpy_mode, cup_rpy_deg in rpy_candidates:
        cup_rotation = rpy_xyz_to_matrix(
            np.radians(np.asarray(cup_rpy_deg, dtype=np.float64))
        )
        for cup in cups:
            rotation = tcp_rotation_for_cup_rotation(cup_rotation, cup)
            rpy_deg = np.degrees(matrix_to_rpy_xyz(rotation))
            current_cup_xyz_mm = current_tcp_xyz_mm + current_rotation @ cup.offset_tool_mm
            selected_offset_base_mm = rotation @ cup.offset_tool_mm
            tcp_approach_xyz_mm = physical_approach_xyz_mm - selected_offset_base_mm
            tcp_pickup_xyz_mm = physical_pickup_xyz_mm - selected_offset_base_mm

            # The candidate-level workspace check is performed on the physical
            # package point before cup compensation.  For the offset secondary
            # cup, the commanded TCP can be hundreds of millimetres away from
            # that point, so validate the actual A/contact TCP targets too.
            if not point_in_workspace(tcp_approach_xyz_mm, args) or not point_in_workspace(
                tcp_pickup_xyz_mm, args
            ):
                print(
                    f"Rejecting suction plan outside configured TCP workspace: "
                    f"cup={cup.name} RPY(deg)={np.asarray(rpy_deg).round(2).tolist()} "
                    f"TCP_A(mm)={tcp_approach_xyz_mm.round(1).tolist()} "
                    f"TCP_pickup(mm)={tcp_pickup_xyz_mm.round(1).tolist()}"
                )
                continue
            travel_mm = float(np.linalg.norm(tcp_approach_xyz_mm - current_tcp_xyz_mm))
            cup_to_package_mm = float(
                np.linalg.norm(physical_pickup_xyz_mm - current_cup_xyz_mm)
            )
            transition_mm = float(np.linalg.norm(tcp_approach_xyz_mm - a_star_xyz_mm))
            radial_reach_mm = float(np.linalg.norm(tcp_approach_xyz_mm))

            try:
                aligned_safe, aligned_d, placement_rotation_deg, _ = aligned_placement_waypoints(
                    candidate, np.asarray(cup_rpy_deg, dtype=np.float64)
                )
            except ValueError as exc:
                print(
                    f"Rejecting suction plan without a valid D long-edge transform: "
                    f"cup={cup.name} mode={rpy_mode}. Reason: {exc}"
                )
                continue
            aligned_safe_tcp = waypoint_for_selected_cup(aligned_safe, cup)
            aligned_d_tcp, _ = placement_tcp_waypoint_for_selected_cup(aligned_d, cup)
            aligned_safe_tcp_xyz_mm = np.asarray(
                [aligned_safe_tcp.x_mm, aligned_safe_tcp.y_mm, aligned_safe_tcp.z_mm],
                dtype=np.float64,
            )
            aligned_d_tcp_xyz_mm = np.asarray(
                [aligned_d_tcp.x_mm, aligned_d_tcp.y_mm, aligned_d_tcp.z_mm],
                dtype=np.float64,
            )
            rotation_safe_tcp_reach_mm = float(np.linalg.norm(aligned_safe_tcp_xyz_mm))
            placement_tcp_reach_mm = float(np.linalg.norm(aligned_d_tcp_xyz_mm))
            placement_max_reach_mm = float(
                getattr(args, "placement_tcp_max_reach_mm", float("inf"))
            )
            placement_y_max_mm = float(
                getattr(args, "placement_tcp_y_max_mm", float("inf"))
            )
            if (
                placement_tcp_reach_mm > placement_max_reach_mm
                or rotation_safe_tcp_reach_mm > placement_max_reach_mm
                or aligned_d_tcp.y_mm > placement_y_max_mm
                or aligned_safe_tcp.y_mm > placement_y_max_mm
            ):
                print(
                    "Rejecting suction plan before pickup: aligned placement moves the active "
                    f"TCP outside the verified envelope; cup={cup.name} mode={rpy_mode} "
                    f"D_rotation={placement_rotation_deg:+.2f}deg "
                    f"D_TCP(mm)={aligned_d_tcp_xyz_mm.round(1).tolist()} "
                    f"D_reach={placement_tcp_reach_mm:.1f}mm "
                    f"safe_TCP(mm)={aligned_safe_tcp_xyz_mm.round(1).tolist()} "
                    f"safe_reach={rotation_safe_tcp_reach_mm:.1f}mm "
                    f"limits=(reach<={placement_max_reach_mm:g}, y<={placement_y_max_mm:g})."
                )
                continue

            unused_clearance_mm = float("inf")
            for other_cup in suction_cup_specs(args):
                if other_cup.name == cup.name:
                    continue
                unused_pickup_xyz_mm = tcp_pickup_xyz_mm + rotation @ other_cup.offset_tool_mm
                other_cup_rotation = rotation @ other_cup.rotation_tool_from_cup
                selected_cup_rotation = rotation @ cup.rotation_tool_from_cup
                contact_normals_parallel = abs(
                    float(selected_cup_rotation[:, 2] @ other_cup_rotation[:, 2])
                ) >= 0.95

                # The old planner excluded the selected package entirely. For
                # parallel cups this allowed the unused second cup/holder to
                # descend into the same large parcel as the active main cup.
                if contact_normals_parallel:
                    target_clearance_mm = candidate_footprint_clearance_mm(
                        candidate,
                        unused_pickup_xyz_mm,
                        cup_collision_radius_mm,
                    )
                    unused_clearance_mm = min(unused_clearance_mm, target_clearance_mm)

                for other_package in other_packages:
                    other_point_mm = np.asarray(other_package.point_base_mm, dtype=np.float64)
                    # Ignore only an obstacle whose top is far below the
                    # unused cup. A taller blocked package must never be
                    # ignored merely because the absolute height difference
                    # is large.
                    if float(unused_pickup_xyz_mm[2] - other_point_mm[2]) > 180.0:
                        continue
                    clearance_mm = candidate_footprint_clearance_mm(
                        other_package,
                        unused_pickup_xyz_mm,
                        cup_collision_radius_mm,
                    )
                    unused_clearance_mm = min(unused_clearance_mm, clearance_mm)

            if np.isfinite(unused_clearance_mm) and unused_clearance_mm < minimum_clearance_mm:
                print(
                    "Rejecting suction plan with unused-cup collision risk: "
                    f"selected_cup={cup.name} RPY(deg)={np.asarray(rpy_deg).round(2).tolist()} "
                    f"edge_clearance={unused_clearance_mm:.1f}mm "
                    f"required={minimum_clearance_mm:.1f}mm."
                )
                continue

            clearance_penalty = 0.0
            if np.isfinite(unused_clearance_mm):
                clearance_penalty = max(0.0, preferred_clearance_mm - unused_clearance_mm) * 8.0

            score = (
                travel_mm
                + cup_to_package_mm
                + 0.35 * transition_mm
                + 0.20 * radial_reach_mm
                + clearance_penalty
                + float(getattr(args, "placement_rotation_score_weight", 10.0))
                * abs(float(placement_rotation_deg))
            )
            plans.append(
                SuctionApproachPlan(
                    cup=cup,
                    approach_xyz_mm=tcp_approach_xyz_mm,
                    pickup_xyz_mm=tcp_pickup_xyz_mm,
                    rpy_deg=np.asarray(rpy_deg, dtype=np.float64),
                    rpy_mode=rpy_mode,
                    score=score,
                    travel_mm=travel_mm,
                    cup_to_package_mm=cup_to_package_mm,
                    radial_reach_mm=radial_reach_mm,
                    unused_cup_clearance_mm=unused_clearance_mm,
                    placement_rotation_deg=float(placement_rotation_deg),
                    placement_tcp_reach_mm=placement_tcp_reach_mm,
                    rotation_safe_tcp_reach_mm=rotation_safe_tcp_reach_mm,
                )
            )

    primary_mode, primary_rpy_deg = rpy_candidates[0]

    def orientation_priority(item: SuctionApproachPlan) -> int:
        # Try both undirected package-long pickup orientations before the old
        # fixed-X fallback. Either aligned branch makes the loaded D rotation
        # approximately zero, while one branch can have much better IK for an
        # offset secondary cup.
        if item.rpy_mode.startswith("align_normal_package_long_for_d"):
            return 0
        if item.rpy_mode == primary_mode:
            return 1
        if item.rpy_mode == "align_normal_fixed_x":
            return 2
        return 3

    plans.sort(
        key=lambda item: (
            orientation_priority(item),
            abs(item.placement_rotation_deg),
            item.score,
        )
    )
    # Controller IK cannot be queried locally on this robot model. Do not
    # truncate the fallback list: a lower-ranked cup/yaw combination can be the
    # only reachable solution near the edge of the real robot workspace.
    return plans


def suction_plan_volume_clear(
    candidate: ClusterCandidate,
    plan: SuctionApproachPlan,
    args: argparse.Namespace,
) -> bool:
    """Run the expensive 3-D check only for a ranked plan about to execute."""
    rotation = rpy_xyz_to_matrix(np.radians(np.asarray(plan.rpy_deg, dtype=np.float64)))
    return pickup_cup_volume_clear(
        candidate,
        plan.pickup_xyz_mm,
        rotation,
        args,
        plan.cup.name,
    )


def waypoint_for_selected_cup(waypoint: RobotWaypoint, cup: SuctionCupSpec) -> RobotWaypoint:
    """Shift TCP so the selected cup reaches a waypoint taught for the primary cup."""
    cup_rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    rotation = tcp_rotation_for_cup_rotation(cup_rotation, cup)
    tcp_shift_mm = rotation @ cup.offset_tool_mm
    tcp_rpy_deg = np.degrees(matrix_to_rpy_xyz(rotation))
    return RobotWaypoint(
        name=f"{waypoint.name}[{cup.name}]",
        x_mm=float(waypoint.x_mm - tcp_shift_mm[0]),
        y_mm=float(waypoint.y_mm - tcp_shift_mm[1]),
        z_mm=float(waypoint.z_mm - tcp_shift_mm[2]),
        rx_deg=float(tcp_rpy_deg[0]),
        ry_deg=float(tcp_rpy_deg[1]),
        rz_deg=float(tcp_rpy_deg[2]),
    )


def waypoint_with_selected_cup_orientation(
    waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
) -> RobotWaypoint:
    """Keep a taught clearance TCP position while orienting the selected cup as taught."""
    cup_rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    tcp_rotation = tcp_rotation_for_cup_rotation(cup_rotation, cup)
    tcp_rpy_deg = np.degrees(matrix_to_rpy_xyz(tcp_rotation))
    return RobotWaypoint(
        name=f"{waypoint.name}[{cup.name}-orientation]",
        x_mm=waypoint.x_mm,
        y_mm=waypoint.y_mm,
        z_mm=waypoint.z_mm,
        rx_deg=float(tcp_rpy_deg[0]),
        ry_deg=float(tcp_rpy_deg[1]),
        rz_deg=float(tcp_rpy_deg[2]),
    )


def loaded_clearance_waypoints_for_cup(
    cup: SuctionCupSpec,
    _current_rpy_deg: np.ndarray | None = None,
) -> tuple[RobotWaypoint, ...]:
    """Return loaded clearance route; side cups intentionally bypass B."""
    if cup.name not in {"third", "fourth"}:
        return WAYPOINT_A_STAR, WAYPOINT_B

    # Reach the complete physically taught side transition pose, then keep its
    # face-down orientation through B. Do not preserve an arbitrary pickup yaw:
    # it can select the wrong J5/J6 wrist branch.
    route_rpy_deg = np.asarray(SIDE_CUP_FACE_DOWN_RPY_DEG, dtype=np.float64)
    side_a = RobotWaypoint(
        name=f"side-loaded-transition[{cup.name}]",
        x_mm=WAYPOINT_SIDE_A_STAR.x_mm,
        y_mm=WAYPOINT_SIDE_A_STAR.y_mm,
        z_mm=WAYPOINT_SIDE_A_STAR.z_mm,
        rx_deg=float(route_rpy_deg[0]),
        ry_deg=float(route_rpy_deg[1]),
        rz_deg=float(route_rpy_deg[2]),
    )
    # B has no functional role for side cups, and every face-down B variant
    # tested on the real robot was rejected with confData -50021. Continue
    # directly from this verified transition to functional C instead.
    return (side_a,)


def move_loaded_clearance_waypoint(
    waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    continuous_rpy_deg: np.ndarray,
):
    """Move a loaded clearance point without allowing a wrist-branch change."""
    return move_pose_with_singularity_fallback(
        robot,
        waypoint.x_mm,
        waypoint.y_mm,
        waypoint.z_mm,
        float(continuous_rpy_deg[0]),
        float(continuous_rpy_deg[1]),
        float(continuous_rpy_deg[2]),
        replace(motion_options, motion="movej", zone_mm=0.0, use_current_conf_data=True),
        allow_clear_confdata_retry=False,
        allow_movej_singularity_retry=False,
    )


def cup_center_at_tcp_waypoint(waypoint: RobotWaypoint, cup: SuctionCupSpec) -> np.ndarray:
    rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    tcp_xyz_mm = np.asarray([waypoint.x_mm, waypoint.y_mm, waypoint.z_mm], dtype=np.float64)
    return tcp_xyz_mm + rotation @ cup.offset_tool_mm


def package_long_axis_base(candidate: ClusterCandidate) -> np.ndarray:
    """Return the undirected package long axis projected into the base XY plane."""
    if candidate.short_axis_base is None:
        raise RuntimeError(
            f"Candidate #{candidate.index} has no calibrated OBB edge direction; "
            "long-edge placement cannot be guaranteed."
        )

    short_axis_xy = np.asarray(candidate.short_axis_base, dtype=np.float64).copy()
    short_axis_xy[2] = 0.0
    short_length = float(np.linalg.norm(short_axis_xy))
    if short_length < 1e-6:
        raise RuntimeError(
            f"Candidate #{candidate.index} has a degenerate base-frame short edge; "
            "long-edge placement cannot be guaranteed."
        )
    short_axis_xy /= short_length

    # Rotating a short-edge direction by +90 degrees produces one of the two
    # equivalent long-edge directions. The later angle calculation is modulo
    # 180 degrees, so the OBB sign ambiguity does not change the result.
    return np.asarray([-short_axis_xy[1], short_axis_xy[0], 0.0], dtype=np.float64)


def wrap_undirected_angle_deg(angle_deg: float) -> float:
    """Wrap an axis-alignment angle to [-90, 90), because long edges are undirected."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def placement_local_z_delta_deg(
    package_long_base: np.ndarray,
    pickup_rotation: np.ndarray,
    placement_reference_rotation: np.ndarray,
) -> float:
    """Compute the smallest local-Z turn that maps the package long edge to D tool Y."""
    long_axis_tool = pickup_rotation.T @ np.asarray(package_long_base, dtype=np.float64)
    tool_xy_length = float(np.linalg.norm(long_axis_tool[:2]))
    if tool_xy_length < 1e-6:
        raise RuntimeError("Package long edge is parallel to the pickup tool Z axis.")

    # At the taught D pose, tool Y is the calibrated platform-long direction.
    # Express that direction in the D reference frame so the implementation
    # remains correct even though the taught D pose has small Rx/Ry offsets.
    platform_long_base = placement_reference_rotation[:, 1]
    desired_tool = placement_reference_rotation.T @ platform_long_base
    current_angle_deg = float(np.degrees(np.arctan2(long_axis_tool[1], long_axis_tool[0])))
    desired_angle_deg = float(np.degrees(np.arctan2(desired_tool[1], desired_tool[0])))
    return wrap_undirected_angle_deg(desired_angle_deg - current_angle_deg)


def rotate_waypoint_about_local_z(
    waypoint: RobotWaypoint,
    local_z_delta_deg: float,
    *,
    name: str | None = None,
) -> RobotWaypoint:
    """Apply a local suction-axis rotation without changing the physical cup-center point."""
    reference_rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    local_z_rotation = rpy_xyz_to_matrix(np.radians([0.0, 0.0, local_z_delta_deg]))
    target_rotation = reference_rotation @ local_z_rotation
    target_rpy_deg = np.degrees(matrix_to_rpy_xyz(target_rotation))
    return RobotWaypoint(
        name=name or f"{waypoint.name}[align-long-edge]",
        x_mm=waypoint.x_mm,
        y_mm=waypoint.y_mm,
        z_mm=waypoint.z_mm,
        rx_deg=float(target_rpy_deg[0]),
        ry_deg=float(target_rpy_deg[1]),
        rz_deg=float(target_rpy_deg[2]),
    )


def local_z_offset_between_waypoints_deg(
    reference_waypoint: RobotWaypoint,
    reached_tcp_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
) -> float:
    """Return the selected cup's local-Z offset from a physical reference pose."""
    reference_rotation = rpy_xyz_to_matrix(
        np.radians(
            [reference_waypoint.rx_deg, reference_waypoint.ry_deg, reference_waypoint.rz_deg]
        )
    )
    reached_tcp_rotation = rpy_xyz_to_matrix(
        np.radians(
            [reached_tcp_waypoint.rx_deg, reached_tcp_waypoint.ry_deg, reached_tcp_waypoint.rz_deg]
        )
    )
    reached_cup_rotation = cup_rotation_at_tcp_rotation(reached_tcp_rotation, cup)
    relative_rotation = reference_rotation.T @ reached_cup_rotation
    return float(np.degrees(np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])))


def nearest_undirected_target_angle_deg(target_deg: float, current_deg: float) -> float:
    """Choose the equivalent long-edge target (modulo 180 deg) nearest current yaw."""
    equivalents = [float(target_deg + 180.0 * turn) for turn in range(-2, 3)]
    return min(equivalents, key=lambda value: abs(value - current_deg))


def angles_toward_fallback_deg(
    target_deg: float,
    fallback_deg: float,
    step_deg: float = 2.0,
) -> list[float]:
    """Return progressively smaller corrections, closest to the target first."""
    distance = float(fallback_deg - target_deg)
    if abs(distance) < 1e-9:
        return []
    count = max(1, int(np.ceil(abs(distance) / step_deg)))
    direction = 1.0 if distance > 0.0 else -1.0
    return [
        float(target_deg + direction * min(step_deg * index, abs(distance)))
        for index in range(1, count + 1)
    ]


def placement_alignment_error_deg(requested_deg: float, actual_deg: float) -> float:
    """Undirected package-long-edge error between requested and actual local yaw."""
    return abs(wrap_undirected_angle_deg(float(actual_deg) - float(requested_deg)))


def verified_placement_angles_for_cup(args: argparse.Namespace, cup_name: str) -> list[float]:
    """Return the independently calibrated D-angle set for one suction cup."""
    cup_specific = getattr(args, f"{cup_name}_verified_placement_angles_deg", None)
    configured = (
        cup_specific
        if cup_specific is not None
        else getattr(args, "verified_placement_angles_deg", (0.0,))
    )
    return [float(angle) for angle in configured]


def verified_placement_fallback_angles_deg(
    requested_deg: float,
    current_deg: float,
    verified_angles_deg: Iterable[float],
) -> list[float]:
    """Return unique, physically verified fallbacks ordered by alignment quality.

    Reachability is not assumed to vary continuously with yaw.  Consequently
    this deliberately does not interpolate unverified two-degree trial poses.
    """
    unique: list[float] = []
    for raw_angle in verified_angles_deg:
        angle = nearest_undirected_target_angle_deg(float(raw_angle), float(current_deg))
        if any(abs(angle - existing) < 1e-6 for existing in unique):
            continue
        unique.append(angle)
    return sorted(
        unique,
        key=lambda angle: (
            placement_alignment_error_deg(requested_deg, angle),
            abs(angle - current_deg),
        ),
    )


def aligned_placement_waypoints(
    candidate: ClusterCandidate,
    pickup_rpy_deg: np.ndarray,
) -> tuple[RobotWaypoint, RobotWaypoint, float, float]:
    """Build safe-rotation and D poses that align package long edge to platform long edge."""
    long_axis_base = package_long_axis_base(candidate)
    pickup_rotation = rpy_xyz_to_matrix(np.radians(np.asarray(pickup_rpy_deg, dtype=np.float64)))
    d_reference_rotation = rpy_xyz_to_matrix(
        np.radians([WAYPOINT_D.rx_deg, WAYPOINT_D.ry_deg, WAYPOINT_D.rz_deg])
    )
    local_z_delta_deg = placement_local_z_delta_deg(
        long_axis_base,
        pickup_rotation,
        d_reference_rotation,
    )
    safe_waypoint = rotate_waypoint_about_local_z(
        WAYPOINT_D_ROTATE_SAFE,
        local_z_delta_deg,
        name="D-rotate-safe[align-long-edge]",
    )
    placement_waypoint = rotate_waypoint_about_local_z(
        WAYPOINT_D,
        local_z_delta_deg,
        name="D[align-long-edge]",
    )

    placement_rotation = rpy_xyz_to_matrix(
        np.radians(
            [
                placement_waypoint.rx_deg,
                placement_waypoint.ry_deg,
                placement_waypoint.rz_deg,
            ]
        )
    )
    long_axis_tool = pickup_rotation.T @ long_axis_base
    predicted_long_base = placement_rotation @ long_axis_tool
    predicted_long_xy = predicted_long_base[:2]
    platform_long_xy = d_reference_rotation[:2, 1].copy()
    predicted_long_xy /= max(1e-9, float(np.linalg.norm(predicted_long_xy)))
    platform_long_xy /= max(1e-9, float(np.linalg.norm(platform_long_xy)))
    alignment_error_deg = float(
        np.degrees(
            np.arccos(
                np.clip(abs(float(np.dot(predicted_long_xy, platform_long_xy))), -1.0, 1.0)
            )
        )
    )
    return safe_waypoint, placement_waypoint, local_z_delta_deg, alignment_error_deg


def placement_tcp_waypoint_for_selected_cup(
    aligned_d_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
) -> tuple[RobotWaypoint, np.ndarray]:
    """Command D so the selected cup center, not the main TCP, reaches D."""
    # D is a functional package-placement point. Do not retain its taught
    # main-TCP XY for an offset cup: that puts the package at a different XY
    # position. Compensate all three axes exactly as at C and the safe point.
    tcp_waypoint = waypoint_for_selected_cup(aligned_d_waypoint, cup)
    selected_cup_center_mm = require_selected_cup_at_functional_waypoint(
        tcp_waypoint,
        aligned_d_waypoint,
        cup,
    )
    return tcp_waypoint, selected_cup_center_mm


def barcode_side_view_waypoint(waypoint: RobotWaypoint) -> RobotWaypoint:
    """Return the taught alternate camera view while keeping the cup center fixed."""
    reference_rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    target_rpy_deg = np.degrees(
        matrix_to_rpy_xyz(reference_rotation @ C_BARCODE_SIDE_VIEW_ROTATION)
    )
    return RobotWaypoint(
        name=f"{waypoint.name}[barcode-side-view]",
        x_mm=waypoint.x_mm,
        y_mm=waypoint.y_mm,
        z_mm=waypoint.z_mm,
        rx_deg=float(target_rpy_deg[0]),
        ry_deg=float(target_rpy_deg[1]),
        rz_deg=float(target_rpy_deg[2]),
    )


def should_inspect_c_side_view(result: WaybillInspectionResult | None) -> bool:
    """Use the fifth-face view only when the initial bottom/back view found no waybill."""
    return result is None or not result.has_waybill


def require_selected_cup_at_functional_waypoint(
    commanded_tcp_waypoint: RobotWaypoint,
    physical_functional_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
    *,
    tolerance_mm: float = 0.5,
) -> np.ndarray:
    actual_center_mm = cup_center_at_tcp_waypoint(commanded_tcp_waypoint, cup)
    expected_center_mm = np.asarray(
        [
            physical_functional_waypoint.x_mm,
            physical_functional_waypoint.y_mm,
            physical_functional_waypoint.z_mm,
        ],
        dtype=np.float64,
    )
    error_mm = float(np.linalg.norm(actual_center_mm - expected_center_mm))
    if error_mm > tolerance_mm:
        raise RuntimeError(
            f"{cup.name} cup-center compensation for {physical_functional_waypoint.name} "
            f"is invalid: expected={expected_center_mm.round(3).tolist()} "
            f"actual={actual_center_mm.round(3).tolist()} error={error_mm:.3f}mm"
        )
    return actual_center_mm


def functional_waypoint_candidates_for_cup(
    physical_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
    current_tcp_xyz_mm: np.ndarray,
    current_tcp_rpy_deg: np.ndarray,
    *,
    prefer_original_orientation: bool,
    allow_yaw_alternatives: bool = True,
) -> list[RobotWaypoint]:
    """Create same-center functional poses with yaw-only IK alternatives."""
    original_tcp_waypoint = waypoint_for_selected_cup(physical_waypoint, cup)
    if not allow_yaw_alternatives or float(np.linalg.norm(cup.offset_tool_mm)) < 1e-6:
        return [original_tcp_waypoint]

    current_rotation = rpy_xyz_to_matrix(
        np.radians(np.asarray(current_tcp_rpy_deg, dtype=np.float64))
    )
    alternatives: list[tuple[float, float, RobotWaypoint]] = []
    original_rotation = rpy_xyz_to_matrix(
        np.radians(
            [
                physical_waypoint.rx_deg,
                physical_waypoint.ry_deg,
                physical_waypoint.rz_deg,
            ]
        )
    )
    for yaw_offset_deg in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 180.0):
        # Rotate around the cup's own Z/contact-normal axis. Unlike replacing
        # Euler RZ directly, this preserves the exact world-space cup normal
        # (and therefore the package bottom plane presented to the C camera).
        local_z_rotation = rpy_xyz_to_matrix(np.radians([0.0, 0.0, yaw_offset_deg]))
        variant_rotation = original_rotation @ local_z_rotation
        variant_rpy_deg = np.degrees(matrix_to_rpy_xyz(variant_rotation))
        yaw_variant = RobotWaypoint(
            name=f"{physical_waypoint.name}[{cup.name},yaw{yaw_offset_deg:+.0f}]",
            x_mm=physical_waypoint.x_mm,
            y_mm=physical_waypoint.y_mm,
            z_mm=physical_waypoint.z_mm,
            rx_deg=float(variant_rpy_deg[0]),
            ry_deg=float(variant_rpy_deg[1]),
            rz_deg=float(variant_rpy_deg[2]),
        )
        tcp_waypoint = waypoint_for_selected_cup(yaw_variant, cup)
        tcp_xyz_mm = np.asarray(
            [tcp_waypoint.x_mm, tcp_waypoint.y_mm, tcp_waypoint.z_mm],
            dtype=np.float64,
        )
        radial_reach_mm = float(np.linalg.norm(tcp_xyz_mm))
        travel_mm = float(np.linalg.norm(tcp_xyz_mm - current_tcp_xyz_mm))
        geometry_score = radial_reach_mm + 0.25 * travel_mm
        orientation_distance_deg = rotation_distance_deg(current_rotation, variant_rotation)
        alternatives.append((orientation_distance_deg, geometry_score, tcp_waypoint))

    if prefer_original_orientation:
        alternatives.sort(key=lambda item: (item[0], item[1]))
        return [original_tcp_waypoint, *(item[2] for item in alternatives)]

    original_xyz_mm = np.asarray(
        [original_tcp_waypoint.x_mm, original_tcp_waypoint.y_mm, original_tcp_waypoint.z_mm],
        dtype=np.float64,
    )
    original_geometry_score = float(np.linalg.norm(original_xyz_mm)) + 0.25 * float(
        np.linalg.norm(original_xyz_mm - current_tcp_xyz_mm)
    )
    original_rotation_distance_deg = rotation_distance_deg(current_rotation, original_rotation)
    # In-plane yaw is functionally free at C, but a 120-degree turn is not a
    # sensible first choice from B. Prefer the shortest orientation change;
    # use reach/travel only as a tie-breaker, and retain all safe fallbacks.
    ranked = [
        (original_rotation_distance_deg, original_geometry_score, original_tcp_waypoint),
        *alternatives,
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def move_selected_cup_to_functional_waypoint(
    physical_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    *,
    prefer_original_orientation: bool,
    allow_yaw_alternatives: bool = True,
    allow_current_conf_movej_fallback: bool = True,
) -> tuple[RobotWaypoint, np.ndarray]:
    current_pose = robot.read_current_pose()
    current_tcp_xyz_mm = current_pose.translation_mm()
    current_tcp_rpy_deg = current_pose.rpy_deg_xyz()
    candidates = functional_waypoint_candidates_for_cup(
        physical_waypoint,
        cup,
        current_tcp_xyz_mm,
        current_tcp_rpy_deg,
        prefer_original_orientation=prefer_original_orientation,
        allow_yaw_alternatives=allow_yaw_alternatives,
    )
    last_error: Exception | None = None
    for rank, tcp_waypoint in enumerate(candidates, start=1):
        cup_center_mm = require_selected_cup_at_functional_waypoint(
            tcp_waypoint,
            physical_waypoint,
            cup,
        )
        tcp_xyz_mm = np.asarray(
            [tcp_waypoint.x_mm, tcp_waypoint.y_mm, tcp_waypoint.z_mm],
            dtype=np.float64,
        )
        print(
            f"Trying functional {physical_waypoint.name} plan #{rank}: cup={cup.name} "
            f"TCP(mm)={tcp_xyz_mm.round(1).tolist()} "
            f"RPY(deg)={[round(tcp_waypoint.rx_deg, 2), round(tcp_waypoint.ry_deg, 2), round(tcp_waypoint.rz_deg, 2)]} "
            f"cup_center(mm)={cup_center_mm.round(1).tolist()} "
            f"reach={float(np.linalg.norm(tcp_xyz_mm)):.1f}mm"
        )
        try:
            move_loaded_transfer_with_singularity_fallback(
                tcp_waypoint,
                robot,
                motion_options,
                allow_current_conf_movej_fallback=allow_current_conf_movej_fallback,
            )
            return tcp_waypoint, cup_center_mm
        except Exception as exc:
            last_error = exc
            if is_operator_software_stop_error(exc) or is_robot_power_or_safety_state_error(exc):
                raise
            retry_reason = None
            if is_no_ik_solution_error(exc):
                retry_reason = "no IK"
            elif is_unsafe_rpy_step_error(exc):
                retry_reason = "unsafe orientation step"
            elif is_path_singularity_error(exc):
                retry_reason = "path singularity"
            if retry_reason is None:
                raise
            print(
                f"Functional {physical_waypoint.name} plan #{rank} rejected ({retry_reason}); "
                "trying the next shorter/safe yaw plan while keeping the same physical center. "
                f"Controller error: {exc}"
            )
    raise RuntimeError(
        f"No reachable {physical_waypoint.name} pose keeps the {cup.name} cup center at the "
        f"required functional point. Last controller error: {last_error}"
    )


def rotate_selected_cup_at_functional_waypoint(
    physical_reference_waypoint: RobotWaypoint,
    local_z_delta_deg: float,
    cup: SuctionCupSpec,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    *,
    start_local_z_deg: float = 0.0,
    max_step_deg: float = 10.0,
    refinement_tolerance_deg: float = 0.5,
) -> tuple[RobotWaypoint, np.ndarray]:
    """Rotate toward the requested angle and retain the closest reachable result."""
    if max_step_deg <= 0.0 or not np.isfinite(max_step_deg):
        raise ValueError("max_step_deg must be a positive finite value.")
    if refinement_tolerance_deg <= 0.0 or not np.isfinite(refinement_tolerance_deg):
        raise ValueError("refinement_tolerance_deg must be a positive finite value.")
    start_delta_deg = float(start_local_z_deg)
    target_delta_deg = float(local_z_delta_deg)
    total_delta_deg = target_delta_deg - start_delta_deg
    step_count = max(1, int(np.ceil(abs(total_delta_deg) / max_step_deg)))
    reached_waypoint: RobotWaypoint | None = None
    cup_center_mm: np.ndarray | None = None
    completed_deltas: list[float] = []
    failed_delta_deg: float | None = None
    try:
        for step_index in range(1, step_count + 1):
            step_delta_deg = start_delta_deg + total_delta_deg * step_index / step_count
            step_waypoint = rotate_waypoint_about_local_z(
                physical_reference_waypoint,
                step_delta_deg,
                name=(
                    f"{physical_reference_waypoint.name}[align-long-edge "
                    f"{step_index}/{step_count}]"
                ),
            )
            print(
                f"Selected-cup rotation step {step_index}/{step_count}: "
                f"cup={cup.name} local-Z={step_delta_deg:+.2f}deg "
                f"(from {start_delta_deg:+.2f} to {target_delta_deg:+.2f}deg)."
            )
            reached_waypoint, cup_center_mm = move_selected_cup_to_functional_waypoint(
                step_waypoint,
                cup,
                robot,
                motion_options,
                prefer_original_orientation=True,
                allow_yaw_alternatives=False,
                allow_current_conf_movej_fallback=False,
            )
            completed_deltas.append(step_delta_deg)
    except Exception as rotation_error:
        if (is_operator_software_stop_error(rotation_error)
                or is_robot_power_or_safety_state_error(rotation_error)):
            raise
        failed_delta_deg = step_delta_deg
        last_reachable_deg = completed_deltas[-1] if completed_deltas else start_delta_deg
        last_reachable_waypoint = rotate_waypoint_about_local_z(
            physical_reference_waypoint,
            last_reachable_deg,
            name=f"{physical_reference_waypoint.name}[last-reachable]",
        )
        print(
            f"Loaded rotation target {failed_delta_deg:+.2f}deg was rejected after "
            f"{last_reachable_deg:+.2f}deg; restoring the last confirmed pose, then "
            f"refining to within {refinement_tolerance_deg:.2f}deg of the reachability boundary."
        )
        try:
            reached_waypoint, cup_center_mm = move_selected_cup_to_functional_waypoint(
                last_reachable_waypoint,
                cup,
                robot,
                motion_options,
                prefer_original_orientation=True,
                allow_yaw_alternatives=False,
                allow_current_conf_movej_fallback=False,
            )
        except Exception as recovery_error:
            raise RuntimeError(
                "Loaded rotation failed and the last confirmed pose could not be restored; "
                "the package orientation is uncertain. "
                f"Rotation error: {rotation_error}; recovery error: {recovery_error}"
            ) from recovery_error

        if not (is_no_ik_solution_error(rotation_error) or is_path_singularity_error(rotation_error)):
            raise SafeRotationRecoveredError(
                f"Loaded rotation failed for a non-searchable reason and was restored to "
                f"{last_reachable_deg:+.2f}deg: {rotation_error}"
            ) from rotation_error

        reachable_deg = last_reachable_deg
        unreachable_deg = failed_delta_deg
        refinement_index = 0
        while abs(unreachable_deg - reachable_deg) > refinement_tolerance_deg:
            refinement_index += 1
            trial_deg = (reachable_deg + unreachable_deg) * 0.5
            trial_waypoint = rotate_waypoint_about_local_z(
                physical_reference_waypoint,
                trial_deg,
                name=f"{physical_reference_waypoint.name}[adaptive-{refinement_index}]",
            )
            print(
                f"Adaptive rotation trial {refinement_index}: cup={cup.name} "
                f"local-Z={trial_deg:+.2f}deg between reachable={reachable_deg:+.2f}deg "
                f"and rejected={unreachable_deg:+.2f}deg."
            )
            try:
                trial_reached, trial_center = move_selected_cup_to_functional_waypoint(
                    trial_waypoint,
                    cup,
                    robot,
                    motion_options,
                    prefer_original_orientation=True,
                    allow_yaw_alternatives=False,
                    allow_current_conf_movej_fallback=False,
                )
            except Exception as trial_error:
                if (is_operator_software_stop_error(trial_error)
                        or is_robot_power_or_safety_state_error(trial_error)):
                    raise
                if not (is_no_ik_solution_error(trial_error) or is_path_singularity_error(trial_error)):
                    raise SafeRotationRecoveredError(
                        f"Adaptive rotation failed for a non-searchable reason at "
                        f"{trial_deg:+.2f}deg: {trial_error}"
                    ) from trial_error
                unreachable_deg = trial_deg
                restore_waypoint = rotate_waypoint_about_local_z(
                    physical_reference_waypoint,
                    reachable_deg,
                    name=f"{physical_reference_waypoint.name}[adaptive-restore]",
                )
                reached_waypoint, cup_center_mm = move_selected_cup_to_functional_waypoint(
                    restore_waypoint,
                    cup,
                    robot,
                    motion_options,
                    prefer_original_orientation=True,
                    allow_yaw_alternatives=False,
                    allow_current_conf_movej_fallback=False,
                )
            else:
                reachable_deg = trial_deg
                reached_waypoint, cup_center_mm = trial_reached, trial_center
        print(
            f"Adaptive rotation accepted the closest reachable angle {reachable_deg:+.2f}deg "
            f"for requested {target_delta_deg:+.2f}deg "
            f"(remaining error={abs(target_delta_deg - reachable_deg):.2f}deg)."
        )
    if reached_waypoint is None or cup_center_mm is None:
        raise RuntimeError("Selected-cup rotation produced no motion step.")
    return reached_waypoint, cup_center_mm


def return_empty_from_d_via_rotation_safe(
    cup: SuctionCupSpec,
    placed_local_z_deg: float,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    on_safe_arrival: Callable[[], None] | None = None,
) -> None:
    """Lift the empty tool from D and unwind its placement yaw at the safe high point."""
    aligned_safe = rotate_waypoint_about_local_z(
        WAYPOINT_D_ROTATE_SAFE,
        placed_local_z_deg,
        name="D-rotate-safe[empty-return]",
    )
    print(
        "Returning empty from D to the high rotation-safe point while preserving "
        f"the placed orientation ({placed_local_z_deg:+.2f}deg)."
    )
    move_selected_cup_to_functional_waypoint(
        aligned_safe,
        cup,
        robot,
        motion_options,
        prefer_original_orientation=True,
        allow_yaw_alternatives=False,
    )
    if on_safe_arrival is not None:
        on_safe_arrival()
    if abs(placed_local_z_deg) >= 0.05:
        print(
            "Unwinding the empty tool at D-rotate-safe before returning through C/B."
        )
        rotate_selected_cup_at_functional_waypoint(
            WAYPOINT_D_ROTATE_SAFE,
            0.0,
            cup,
            robot,
            motion_options,
            start_local_z_deg=placed_local_z_deg,
        )


def is_no_ik_solution_error(exc: Exception) -> bool:
    message = str(exc)
    return "-50021" in message or "50021" in message


def is_unsafe_rpy_step_error(exc: Exception) -> bool:
    return "Unsafe RPY step" in str(exc)


def is_path_singularity_error(exc: Exception) -> bool:
    message = str(exc)
    return "-50102" in message or "50102" in message or "奇异点" in message


def compute_approach_geometry(
    candidate: ClusterCandidate,
    robot: XCoreRobotClient,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[tuple[str, np.ndarray]]]:
    current_pose = robot.read_current_pose()
    target_rpy_deg, rpy_mode = target_rpy_for_candidate(candidate, current_pose, args)
    surface_normal = np.asarray(candidate.normal_base, dtype=np.float64)
    normal_length = float(np.linalg.norm(surface_normal))
    if normal_length < 1e-9:
        raise RuntimeError(f"Candidate #{candidate.index} has an invalid surface normal.")
    surface_normal /= normal_length

    approach_xyz_mm = target_point_for_candidate(
        candidate.point_base_mm,
        surface_normal,
        args.standoff_mm,
        args.standoff_mode,
    )
    pickup_xyz_mm = approach_xyz_mm - args.pickup_down_mm * surface_normal
    rpy_candidates = target_rpy_candidates_for_candidate(
        candidate,
        current_pose,
        args,
        target_rpy_deg,
        rpy_mode,
    )
    return approach_xyz_mm, pickup_xyz_mm, target_rpy_deg, rpy_mode, rpy_candidates


def move_approach_down_to_pickup(
    robot: XCoreRobotClient,
    approach_xyz_mm: np.ndarray,
    pickup_xyz_mm: np.ndarray,
    target_rpy_deg: np.ndarray,
    linear_options: MotionOptions,
):
    """Validate a suction plan through its pickup endpoint while suction is off."""
    descent_start_pose = robot.read_current_pose()
    print(
        f"Descending A -> pickup by {float(np.linalg.norm(pickup_xyz_mm - approach_xyz_mm)):.1f} mm: "
        f"start XYZ(mm)={descent_start_pose.translation_mm().round(1).tolist()} "
        f"target XYZ(mm)={pickup_xyz_mm.round(1).tolist()}"
    )
    pickup_pose = move_pose_with_singularity_fallback(
        robot,
        float(pickup_xyz_mm[0]),
        float(pickup_xyz_mm[1]),
        float(pickup_xyz_mm[2]),
        float(target_rpy_deg[0]),
        float(target_rpy_deg[1]),
        float(target_rpy_deg[2]),
        linear_options,
        allow_clear_confdata_retry=False,
        allow_movej_singularity_retry=False,
    )
    pickup_error_mm = float(np.linalg.norm(pickup_pose.translation_mm() - pickup_xyz_mm))
    actual_descent_mm = float(
        np.linalg.norm(pickup_pose.translation_mm() - descent_start_pose.translation_mm())
    )
    print(
        f"Pickup descent complete: actual TCP travel={actual_descent_mm:.1f} mm "
        f"final XYZ(mm)={pickup_pose.translation_mm().round(1).tolist()} "
        f"target_error={pickup_error_mm:.1f} mm"
    )
    return pickup_pose


def execute_candidate_sequence(
    candidate: ClusterCandidate,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    args: argparse.Namespace,
    next_candidate: ClusterCandidate | None = None,
    prepositioned_approach: ApproachPlan | None = None,
    waybill_inspector: AsyncWaybillInspector | None = None,
    scene_candidates: list[ClusterCandidate] | None = None,
    roi_config: RoiConfig | None = None,
) -> tuple[bool, ApproachPlan | None, bool]:
    if not candidate.motion_safe:
        print(f"Candidate #{candidate.index} is blocked for batch motion: {candidate.filter_note}")
        return False, None, False

    # Long-edge alignment is mandatory for a completed package. Validate the
    # detected axis before suction so a missing/degenerate OBB cannot leave us
    # holding a package whose required placement orientation is unknown.
    try:
        detected_long_axis_base = package_long_axis_base(candidate)
    except Exception as exc:
        print(
            f"Candidate #{candidate.index} cannot satisfy long-edge placement; "
            f"skipping it before suction. Reason: {exc}"
        )
        return True, None, False
    print(
        f"Candidate #{candidate.index} package long edge (base XY)="
        f"{detected_long_axis_base.round(4).tolist()}; "
        "target at D is parallel to the taught D tool Y axis."
    )

    try:
        approach_xyz_mm, pickup_xyz_mm, target_rpy_deg, rpy_mode, rpy_candidates = compute_approach_geometry(
            candidate,
            robot,
            args,
        )
    except Exception as exc:
        print(f"Candidate #{candidate.index} approach planning failed: {exc}")
        return False, None, False

    print(
        f"Starting batch candidate #{candidate.index}: "
        f"type={candidate.class_name} "
        f"A(mm)={approach_xyz_mm.round(1).tolist()} "
        f"pickup(mm)={pickup_xyz_mm.round(1).tolist()} "
        f"RPY(deg)={target_rpy_deg.round(2).tolist()} "
        f"rpy_mode={rpy_mode}"
    )

    try:
        approach_options = replace(
            motion_options,
            motion="movej",
            use_current_conf_data=True,
        )
        linear_options = replace(
            motion_options,
            motion="movel",
            use_current_conf_data=True,
        )
        stop_options = replace(linear_options, zone_mm=0.0)
        pass_through_zone_mm = max(0.0, float(args.pass_through_zone_mm))

        print(
            "Pickup orientation: first align tool Y with the detected package long edge so "
            "the loaded rotation required at D is approximately zero. Both 180-degree-equivalent "
            f"branches are tried before the legacy fixed-X RZ={FIXED_GRASP_X_YAW_DEG:.2f} deg "
            "and yaw-only IK fallbacks; the suction face remains horizontal."
        )

        suction_active = False
        selected_suction_name = "primary"
        selected_suction_port = int(args.suction_do_port)
        selected_suction_cup = suction_cup_specs(args)[0]
        selected_rpy_deg: np.ndarray | None = None
        selected_rpy_mode = rpy_mode
        last_approach_error: Exception | None = None
        pickup_pose = None
        if prepositioned_approach is not None and prepositioned_approach.candidate_index == candidate.index:
            approach_xyz_mm = prepositioned_approach.approach_xyz_mm
            pickup_xyz_mm = prepositioned_approach.pickup_xyz_mm
            selected_rpy_deg = prepositioned_approach.rpy_deg
            selected_rpy_mode = prepositioned_approach.rpy_mode
            selected_suction_name = prepositioned_approach.suction_name
            selected_suction_port = prepositioned_approach.suction_do_port
            selected_suction_cup = next(
                (
                    cup
                    for cup in suction_cup_specs(args)
                    if cup.do_port == selected_suction_port
                ),
                suction_cup_specs(args)[0],
            )
            print(
                f"Already at candidate #{candidate.index} A point from previous return path; "
                f"using preselected RPY(deg)={selected_rpy_deg.round(2).tolist()}."
            )
        else:
            current_pose = robot.read_current_pose()
            if pose_is_at_waypoint(current_pose, WAYPOINT_B):
                print(
                    "Empty tool is already at B; using the actual B pose as the "
                    "pickup planning origin."
                )
            else:
                print(
                    "Routing empty tool to verified A* and planning directly from A* "
                    "to dynamic A; the redundant A* -> B -> A* detour is omitted."
                )
                # A previous perpendicular-cup attempt can leave the tool in a
                # side-facing orientation. A* is the physically verified point
                # that normalizes both position and orientation before entering
                # the bin; B is not required for this empty-tool approach.
                if pose_is_at_waypoint(current_pose, WAYPOINT_A_STAR):
                    print("Empty tool is already at A*; skipping the redundant A* command.")
                else:
                    current_rpy_deg = current_pose.rpy_deg_xyz()
                    a_star_raw_rpy_deg = np.asarray(
                        [WAYPOINT_A_STAR.rx_deg, WAYPOINT_A_STAR.ry_deg, WAYPOINT_A_STAR.rz_deg],
                        dtype=np.float64,
                    )
                    a_star_continuous_rpy_deg = unwrap_rpy_deg(
                        a_star_raw_rpy_deg,
                        current_rpy_deg,
                    )
                    try:
                        require_safe_rpy_step(
                            current_rpy_deg,
                            a_star_continuous_rpy_deg,
                            label="current pose -> waypoint A*",
                        )
                    except RuntimeError as direct_a_star_error:
                        print(
                            "Direct empty-tool route to A* exceeds the verified orientation "
                            "step; routing current pose -> B -> A* instead. This is a one-way "
                            f"safety bridge, not an A* -> B -> A* detour. Reason: {direct_a_star_error}"
                        )
                        try:
                            move_waypoint(WAYPOINT_B, robot, approach_options)
                        except RuntimeError as direct_b_error:
                            if not is_unsafe_rpy_step_error(direct_b_error):
                                raise
                            print(
                                "Direct empty-tool route to B also exceeds the verified orientation "
                                "step; using the D restart recovery route. "
                                f"Reason: {direct_b_error}"
                            )
                            recover_empty_tool_from_d_to_b(robot, approach_options)
                    move_waypoint(WAYPOINT_A_STAR, robot, approach_options)
            # Candidate orientation sign/rotation limits must be evaluated from
            # the actual normalized planning pose (B during normal cycles, A*
            # during startup/recovery), never from the stale pose seen before
            # routing to that point.
            (
                approach_xyz_mm,
                pickup_xyz_mm,
                target_rpy_deg,
                rpy_mode,
                rpy_candidates,
            ) = compute_approach_geometry(candidate, robot, args)
            print(
                "Replanned pickup orientation from the actual normalized pose: "
                f"RPY(deg)={target_rpy_deg.round(2).tolist()} mode={rpy_mode}"
            )
            # Rank travel and cup-to-package distances from the same actual pose
            # used for orientation replanning. Reading it before normalization
            # produces stale and misleading dual-cup scores.
            selection_origin_pose = robot.read_current_pose()
            selection_origin_xyz_mm = selection_origin_pose.translation_mm()
            selection_origin_rpy_deg = selection_origin_pose.rpy_deg_xyz()
            suction_plans = build_suction_approach_plans(
                candidate,
                scene_candidates or [candidate],
                selection_origin_xyz_mm,
                selection_origin_rpy_deg,
                approach_xyz_mm,
                pickup_xyz_mm,
                rpy_candidates,
                args,
                roi_config,
            )
            for rank, suction_plan in enumerate(suction_plans, start=1):
                print(
                    f"Selected ranked plan #{rank} for final 3-D cup-volume check: "
                    f"candidate=#{candidate.index} cup={suction_plan.cup.name}."
                )
                if not suction_plan_volume_clear(candidate, suction_plan, args):
                    print(
                        f"Ranked plan #{rank} failed final 3-D cup-volume check; "
                        "checking the next ranked plan."
                    )
                    continue
                try:
                    clearance_text = (
                        "clear"
                        if not np.isfinite(suction_plan.unused_cup_clearance_mm)
                        else f"{suction_plan.unused_cup_clearance_mm:.1f}mm"
                    )
                    print(
                        f"Trying multi-suction plan #{rank}: cup={suction_plan.cup.name} "
                        f"DO{args.suction_do_board}_{suction_plan.cup.do_port} "
                        f"TCP_A(mm)={suction_plan.approach_xyz_mm.round(1).tolist()} "
                        f"RPY(deg)={suction_plan.rpy_deg.round(2).tolist()} "
                        f"travel={suction_plan.travel_mm:.1f}mm "
                        f"cup_to_package={suction_plan.cup_to_package_mm:.1f}mm "
                        f"reach={suction_plan.radial_reach_mm:.1f}mm "
                        f"D_rotation={suction_plan.placement_rotation_deg:+.2f}deg "
                        f"D_TCP_reach={suction_plan.placement_tcp_reach_mm:.1f}mm "
                        f"safe_TCP_reach={suction_plan.rotation_safe_tcp_reach_mm:.1f}mm "
                        f"unused_clearance={clearance_text} score={suction_plan.score:.1f}"
                    )
                    commanded_rpy_deg = move_empty_approach_to_a(
                        robot,
                        suction_plan.approach_xyz_mm,
                        suction_plan.rpy_deg,
                        approach_options,
                    )
                    approach_xyz_mm = suction_plan.approach_xyz_mm
                    pickup_xyz_mm = suction_plan.pickup_xyz_mm
                    # Reuse the continuous equivalent at pickup so the short
                    # MoveL descent cannot numerically wrap back by 360 deg.
                    selected_rpy_deg = commanded_rpy_deg
                    selected_rpy_mode = suction_plan.rpy_mode
                    selected_suction_name = suction_plan.cup.name
                    selected_suction_port = suction_plan.cup.do_port
                    selected_suction_cup = suction_plan.cup
                    # Reaching A alone does not prove that the vertical contact
                    # endpoint is reachable. Validate the complete pre-suction
                    # path before accepting this cup/orientation plan.
                    pickup_pose = move_approach_down_to_pickup(
                        robot,
                        approach_xyz_mm,
                        pickup_xyz_mm,
                        selected_rpy_deg,
                        linear_options,
                    )
                    break
                except Exception as exc:
                    last_approach_error = exc
                    selected_rpy_deg = None
                    pickup_pose = None
                    if suction_active:
                        raise
                    if is_operator_software_stop_error(exc):
                        raise RuntimeError(
                            "Operator software stop is latched; aborting the complete batch "
                            "without trying another suction plan or package."
                        ) from exc
                    if is_robot_power_or_safety_state_error(exc):
                        raise RuntimeError(
                            "Robot power/safety state no longer permits motion; aborting all "
                            "remaining suction/orientation plans instead of repeatedly commanding it. "
                            f"Controller error: {exc}"
                        ) from exc
                    # Clear the rejected/partial motion before routing back
                    # through A* for the next independently reachable plan.
                    robot.stop_motion()
                    print(
                        f"Multi-suction plan #{rank} ({suction_plan.cup.name}) failed before suction; "
                        "trying the next cup/pose plan. "
                        f"Controller error: {exc}"
                    )

        if selected_rpy_deg is None:
            # Every attempt above happens with suction off.  A package near a
            # reach or singularity boundary must not abort the remaining
            # independently reachable packages in the batch.
            print(
                f"Candidate #{candidate.index} has no accepted approach in any allowed orientation; "
                "skipping this package and continuing with the next one. "
                f"Last controller error: {last_approach_error}"
            )
            return True, None, False

        target_rpy_deg = selected_rpy_deg
        rpy_mode = selected_rpy_mode
        print(
            f"Selected pickup orientation {rpy_mode}: "
            f"RPY(deg)={target_rpy_deg.round(2).tolist()}; "
            f"selected cup={selected_suction_name} DO{args.suction_do_board}_{selected_suction_port}"
        )

        # A prepositioned next candidate has already validated only its A point;
        # validate its pickup endpoint here. Normal plans did this in the loop.
        if pickup_pose is None:
            try:
                pickup_pose = move_approach_down_to_pickup(
                    robot,
                    approach_xyz_mm,
                    pickup_xyz_mm,
                    target_rpy_deg,
                    linear_options,
                )
            except Exception as exc:
                print(
                    f"Candidate #{candidate.index} prepositioned pickup endpoint is unreachable; "
                    "skipping this package and continuing with the next one. "
                    f"Controller error: {exc}"
                )
                return True, None, False
        print(
            f"At pickup point for candidate #{candidate.index}; "
            f"enabling {selected_suction_name} suction "
            f"DO{args.suction_do_board}_{selected_suction_port}. "
            f"Waiting {args.pickup_dwell_s:.1f}s."
        )
        # Enforce mutual exclusion even if a previous interrupted cycle left
        # the other valve active.
        set_all_suction_outputs(robot, args, False)
        set_suction_output(robot, args, True, do_port=selected_suction_port)
        suction_active = not args.disable_suction_io
        wait_with_operator_stop(args.pickup_dwell_s, robot, linear_options)

        if selected_suction_cup.name in {"third", "fourth"}:
            print(
                "Retracting vertically to A with the pickup orientation, then routing "
                "through the side loaded transition directly to functional C; B is bypassed."
            )
        else:
            print(
                "Retracting vertically to A with the pickup orientation, then routing "
                "through A* to B with continuous-RPY MoveJ commands."
            )

        # Only the loaded-tool lift must be Cartesian-linear.  Previously A,
        # A* and B were submitted as one MoveL path.  A pickup Rx near +180
        # followed by the equivalent fixed A* Rx near -180 could therefore be
        # interpolated numerically as an almost 360-degree wrist rotation.
        lifted_pose = robot.move_to_pose_mm_deg(
            float(approach_xyz_mm[0]),
            float(approach_xyz_mm[1]),
            float(approach_xyz_mm[2]),
            float(target_rpy_deg[0]),
            float(target_rpy_deg[1]),
            float(target_rpy_deg[2]),
            options=replace(linear_options, zone_mm=0.0, use_current_conf_data=True),
        )

        previous_rpy_deg = lifted_pose.rpy_deg_xyz()
        # Third/fourth cups use their physically verified side A* and retain
        # that pose orientation through B. Primary/secondary keep the original
        # taught A*/B path. Functional C/D below perform their own selected-cup
        # center/orientation compensation.
        loaded_clearance_waypoints = loaded_clearance_waypoints_for_cup(
            selected_suction_cup,
            previous_rpy_deg,
        )
        for waypoint in loaded_clearance_waypoints:
            selected_cup_center_mm = cup_center_at_tcp_waypoint(
                waypoint,
                selected_suction_cup,
            )
            raw_rpy_deg = np.asarray(
                [waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg],
                dtype=np.float64,
            )
            continuous_rpy_deg = unwrap_rpy_deg(raw_rpy_deg, previous_rpy_deg)
            require_safe_rpy_step(
                previous_rpy_deg,
                continuous_rpy_deg,
                label=f"loaded return waypoint {waypoint.name} for {selected_suction_name}",
            )
            print(
                f"Moving loaded return waypoint {waypoint.name} for "
                f"{selected_suction_name}: "
                f"XYZ(mm)={[waypoint.x_mm, waypoint.y_mm, waypoint.z_mm]} "
                f"selected_cup_center(mm)={selected_cup_center_mm.round(1).tolist()} "
                f"raw RPY(deg)={raw_rpy_deg.round(2).tolist()} "
                f"continuous RPY(deg)={continuous_rpy_deg.round(2).tolist()} motion=movej"
            )
            reached_pose = move_loaded_clearance_waypoint(
                waypoint,
                selected_suction_cup,
                robot,
                approach_options,
                continuous_rpy_deg,
            )
            previous_rpy_deg = reached_pose.rpy_deg_xyz()
        side_cup_direct_to_c = selected_suction_cup.name in {"third", "fourth"}
        if waybill_inspector is not None:
            print(
                f"At {'side loaded transition' if side_cup_direct_to_c else 'B'}; "
                f"waiting {args.waybill_start_delay_s:.2f}s before starting "
                "the bottom-camera clip "
                f"(safety limit {args.waybill_capture_duration_s:.2f}s)."
            )
            wait_with_operator_stop(args.waybill_start_delay_s, robot, stop_options)
            waybill_inspector.begin_capture(candidate.index)
            print(
                f"Bottom-camera clip started for candidate #{candidate.index}; "
                "capture continues asynchronously while travelling to C."
            )

        print(
            f"Functional C/D routing for {selected_suction_name} "
            f"DO{args.suction_do_board}_{selected_suction_port}: "
            "using the selected cup's calibrated camera/placement geometry."
        )
        barcode_rotation_options = replace(
            approach_options,
            motion="movej",
            zone_mm=0.0,
            use_current_conf_data=False,
        )
        if selected_suction_cup.name in {"primary", "secondary"}:
            loaded_waypoint_c = WAYPOINT_C_BARCODE
            move_loaded_transfer_with_singularity_fallback(
                loaded_waypoint_c,
                robot,
                barcode_rotation_options,
            )
            selected_cup_center_c_mm = cup_center_at_tcp_waypoint(
                loaded_waypoint_c,
                selected_suction_cup,
            )
            print(
                f"{selected_suction_name.capitalize()} suction uses the exact C TCP pose verified by "
                "run_c_cup_z_rotation_test.cmd --taught-side."
            )
        else:
            loaded_waypoint_c, selected_cup_center_c_mm = move_selected_cup_to_functional_waypoint(
                WAYPOINT_C,
                selected_suction_cup,
                robot,
                approach_options if side_cup_direct_to_c else linear_options,
                prefer_original_orientation=False,
                allow_yaw_alternatives=True,
            )
        if waybill_inspector is not None:
            waybill_inspector.mark_c_arrival(candidate.index)
            required_c_hold_s = (
                float(args.waybill_c_settle_s)
                + float(args.waybill_post_c_capture_s)
                + 0.1
            )
            actual_c_hold_s = max(float(args.waybill_c_dwell_s), required_c_hold_s)
            print(
                f"At functional C with {selected_suction_name} cup/package center="
                f"{selected_cup_center_c_mm.round(1).tolist()}; "
                f"settling for {args.waybill_c_settle_s:.2f}s, then capturing "
                f"{args.waybill_post_c_capture_s:.2f}s of stationary frames; "
                f"total hold={actual_c_hold_s:.2f}s."
            )
            wait_with_operator_stop(actual_c_hold_s, robot, stop_options)
            try:
                c_result = waybill_inspector.wait_for_result(
                    candidate.index,
                    args.waybill_result_timeout_s,
                )
            except Exception as inspection_error:
                print(
                    f"Initial C-view recognition did not return a usable waybill result for candidate "
                    f"#{candidate.index}: {inspection_error}. Trying the taught side view."
                )
                c_result = None

            if should_inspect_c_side_view(c_result):
                if selected_suction_cup.name in {"primary", "secondary"}:
                    # This is already an active tool4 TCP pose. Do not apply
                    # a selected-cup offset a second time.
                    side_view_waypoint = WAYPOINT_C_BARCODE_SIDE
                else:
                    initial_c_tcp_rotation = rpy_xyz_to_matrix(
                        np.radians(
                            [
                                loaded_waypoint_c.rx_deg,
                                loaded_waypoint_c.ry_deg,
                                loaded_waypoint_c.rz_deg,
                            ]
                        )
                    )
                    initial_c_cup_rotation = cup_rotation_at_tcp_rotation(
                        initial_c_tcp_rotation,
                        selected_suction_cup,
                    )
                    initial_c_cup_rpy_deg = np.degrees(
                        matrix_to_rpy_xyz(initial_c_cup_rotation)
                    )
                    initial_c_cup_waypoint = RobotWaypoint(
                        name="C[reached-initial-view]",
                        x_mm=WAYPOINT_C.x_mm,
                        y_mm=WAYPOINT_C.y_mm,
                        z_mm=WAYPOINT_C.z_mm,
                        rx_deg=float(initial_c_cup_rpy_deg[0]),
                        ry_deg=float(initial_c_cup_rpy_deg[1]),
                        rz_deg=float(initial_c_cup_rpy_deg[2]),
                    )
                    side_view_waypoint = barcode_side_view_waypoint(initial_c_cup_waypoint)
                side_view_options = replace(
                    approach_options,
                    motion="movej",
                    zone_mm=0.0,
                    # This matches the verified C-point side-view test: allow
                    # the controller to select the reachable joint branch.
                    use_current_conf_data=False,
                )
                print(
                    "No waybill on the package bottom or back in the initial C camera view; "
                    "rotating 90 degrees to the taught side-view pose for one additional pass."
                )
                moved_to_side_view = False
                try:
                    if selected_suction_cup.name in {"primary", "secondary"}:
                        side_tcp = side_view_waypoint
                        move_loaded_transfer_with_singularity_fallback(
                            side_tcp,
                            robot,
                            side_view_options,
                        )
                        side_center_mm = cup_center_at_tcp_waypoint(
                            side_tcp,
                            selected_suction_cup,
                        )
                    else:
                        side_tcp, side_center_mm = move_selected_cup_to_functional_waypoint(
                            side_view_waypoint,
                            selected_suction_cup,
                            robot,
                            side_view_options,
                            prefer_original_orientation=True,
                            allow_yaw_alternatives=False,
                        )
                    moved_to_side_view = True
                    print(
                        f"At C taught side view: TCP(mm)="
                        f"{[round(side_tcp.x_mm, 1), round(side_tcp.y_mm, 1), round(side_tcp.z_mm, 1)]} "
                        f"cup/package center={side_center_mm.round(1).tolist()}."
                    )
                    waybill_inspector.begin_capture(candidate.index)
                    waybill_inspector.mark_c_arrival(candidate.index)
                    wait_with_operator_stop(actual_c_hold_s, robot, side_view_options)
                    try:
                        side_result = waybill_inspector.wait_for_result(
                            candidate.index,
                            args.waybill_result_timeout_s,
                        )
                        if side_result.has_waybill:
                            barcode_suffix = (
                                f" Barcode: {side_result.barcode}."
                                if side_result.barcode
                                else " Barcode was not decoded."
                            )
                            print(
                                "Waybill detected on the package side at C."
                                f"{barcode_suffix}"
                            )
                        else:
                            print(
                                "No waybill detected on the package side at C; "
                                "returning to the initial C pose and continuing."
                            )
                    except Exception as inspection_error:
                        print(
                            f"C side-view recognition failed or timed out: {inspection_error}. "
                            "Returning to the initial C pose and continuing."
                        )
                except Exception as side_view_error:
                    # Side inspection is optional. An IK failure before the
                    # side motion must not leave a successfully held package
                    # suspended at C or prevent its normal D placement.
                    print(
                        f"C taught side view is unreachable: {side_view_error}. "
                        "Keeping the initial C pose and continuing to D."
                    )
                finally:
                    if moved_to_side_view:
                        print("Returning from C side view to the initial C pose.")
                        move_loaded_transfer_with_singularity_fallback(
                            loaded_waypoint_c,
                            robot,
                            side_view_options,
                        )
            elif c_result is not None:
                barcode_suffix = (
                    f" Barcode: {c_result.barcode}."
                    if c_result.barcode
                    else " Barcode was not decoded."
                )
                print(
                    "Waybill detected on the package bottom/back at the initial C pose; "
                    "the side-view rotation is not required."
                    f"{barcode_suffix}"
                )

        pickup_tcp_rotation = rpy_xyz_to_matrix(np.radians(pickup_pose.rpy_deg_xyz()))
        pickup_cup_rpy_deg = np.degrees(
            matrix_to_rpy_xyz(
                cup_rotation_at_tcp_rotation(
                    pickup_tcp_rotation,
                    selected_suction_cup,
                )
            )
        )
        (
            aligned_rotate_safe,
            aligned_waypoint_d,
            placement_delta_deg,
            predicted_alignment_error_deg,
        ) = aligned_placement_waypoints(candidate, pickup_cup_rpy_deg)
        print(
            f"Long-edge placement plan for candidate #{candidate.index}: "
            f"local suction-axis turn={placement_delta_deg:+.2f}deg, "
            f"predicted platform alignment error={predicted_alignment_error_deg:.3f}deg."
        )
        print(
            "Moving from C to the verified high rotation point first; the calculated "
            "local-Z turn is then executed around the selected physical cup center."
        )
        reached_safe_tcp, _reached_safe_cup_center = move_selected_cup_to_functional_waypoint(
            WAYPOINT_D_ROTATE_SAFE,
            selected_suction_cup,
            robot,
            stop_options,
            prefer_original_orientation=True,
            # Side cups often leave C on a different wrist branch (for example
            # yaw+180). Preserve the physical high-point center while trying
            # equivalent in-plane orientations instead of forcing one IK pose.
            allow_yaw_alternatives=True,
        )
        reached_safe_yaw_deg = local_z_offset_between_waypoints_deg(
            WAYPOINT_D_ROTATE_SAFE,
            reached_safe_tcp,
            selected_suction_cup,
        )
        effective_placement_delta_deg = nearest_undirected_target_angle_deg(
            placement_delta_deg,
            reached_safe_yaw_deg,
        )
        if abs(effective_placement_delta_deg - placement_delta_deg) >= 0.05:
            print(
                f"D-rotate-safe reached with local yaw={reached_safe_yaw_deg:+.2f}deg; "
                f"using equivalent long-edge target={effective_placement_delta_deg:+.2f}deg "
                f"instead of {placement_delta_deg:+.2f}deg."
            )
        placement_delta_deg = effective_placement_delta_deg
        requested_placement_delta_deg = placement_delta_deg
        aligned_rotate_safe = rotate_waypoint_about_local_z(
            WAYPOINT_D_ROTATE_SAFE,
            placement_delta_deg,
            name="D-rotate-safe[align-long-edge]",
        )
        aligned_waypoint_d = rotate_waypoint_about_local_z(
            WAYPOINT_D,
            placement_delta_deg,
            name="D[align-long-edge]",
        )
        if abs(placement_delta_deg - reached_safe_yaw_deg) >= 0.05:
            print(
                "Rotating package at the verified high point from "
                f"{reached_safe_yaw_deg:+.2f}deg to {placement_delta_deg:+.2f}deg "
                "about the selected suction axis."
            )
            try:
                rotated_safe_tcp, _rotated_safe_cup_center = rotate_selected_cup_at_functional_waypoint(
                    WAYPOINT_D_ROTATE_SAFE,
                    placement_delta_deg,
                    selected_suction_cup,
                    robot,
                    stop_options,
                    start_local_z_deg=reached_safe_yaw_deg,
                )
            except SafeRotationRecoveredError as rotation_error:
                print(
                    "Optional long-edge rotation could not complete, but the package was "
                    "restored to the high-point start orientation. Falling back directly "
                    f"to the taught D pose. Reason: {rotation_error}"
                )
                placement_delta_deg = reached_safe_yaw_deg
                aligned_rotate_safe = rotate_waypoint_about_local_z(
                    WAYPOINT_D_ROTATE_SAFE,
                    reached_safe_yaw_deg,
                    name="D-rotate-safe[recovered]",
                )
                # The reached high-point start is an equivalent functional
                # orientation. Use the corresponding D orientation so the
                # following move starts from exactly the recovered yaw.
                aligned_waypoint_d = rotate_waypoint_about_local_z(
                    WAYPOINT_D,
                    reached_safe_yaw_deg,
                    name="D[recovered-fallback]",
                )
                rotated_safe_tcp = reached_safe_tcp
            achieved_placement_delta_deg = local_z_offset_between_waypoints_deg(
                WAYPOINT_D_ROTATE_SAFE,
                rotated_safe_tcp,
                selected_suction_cup,
            )
            if abs(achieved_placement_delta_deg - placement_delta_deg) >= 0.05:
                print(
                    f"Using closest reachable placement rotation "
                    f"{achieved_placement_delta_deg:+.2f}deg instead of requested "
                    f"{placement_delta_deg:+.2f}deg; rebuilding the D target from the "
                    "actual confirmed safe-point orientation."
                )
            placement_delta_deg = achieved_placement_delta_deg
            aligned_rotate_safe = rotate_waypoint_about_local_z(
                WAYPOINT_D_ROTATE_SAFE,
                placement_delta_deg,
                name="D-rotate-safe[closest-reachable]",
            )
            aligned_waypoint_d = rotate_waypoint_about_local_z(
                WAYPOINT_D,
                placement_delta_deg,
                name="D[closest-reachable]",
            )
            expected_safe_cup_rotation = rpy_xyz_to_matrix(
                np.radians(
                    [
                        aligned_rotate_safe.rx_deg,
                        aligned_rotate_safe.ry_deg,
                        aligned_rotate_safe.rz_deg,
                    ]
                )
            )
            expected_safe_rotation = tcp_rotation_for_cup_rotation(
                expected_safe_cup_rotation,
                selected_suction_cup,
            )
            reached_safe_rotation = rpy_xyz_to_matrix(
                np.radians(
                    [
                        rotated_safe_tcp.rx_deg,
                        rotated_safe_tcp.ry_deg,
                        rotated_safe_tcp.rz_deg,
                    ]
                )
            )
            if not np.allclose(reached_safe_rotation, expected_safe_rotation, atol=1e-7):
                raise RuntimeError("Segmented safe-point rotation did not reach the planned orientation.")
        print("Moving from the aligned high point to D without restoring the taught D yaw.")
        loaded_waypoint_d, selected_cup_center_d_mm = placement_tcp_waypoint_for_selected_cup(
            aligned_waypoint_d,
            selected_suction_cup,
        )
        print(
            f"Placement D compensates the full TCP offset for {selected_suction_name}: "
            f"TCP(mm)={[round(loaded_waypoint_d.x_mm, 1), round(loaded_waypoint_d.y_mm, 1), round(loaded_waypoint_d.z_mm, 1)]} "
            f"selected_cup_center(mm)={selected_cup_center_d_mm.round(1).tolist()}."
        )
        placed_local_z_deg = placement_delta_deg
        reached_d = False
        try:
            move_loaded_transfer_with_singularity_fallback(
                loaded_waypoint_d,
                robot,
                stop_options,
            )
            reached_d = True
        except Exception as aligned_d_exc:
            if is_operator_software_stop_error(aligned_d_exc) or is_robot_power_or_safety_state_error(
                aligned_d_exc
            ):
                raise
            if not (
                is_no_ik_solution_error(aligned_d_exc)
                or is_path_singularity_error(aligned_d_exc)
            ):
                raise
            print(
                f"Aligned placement D is unreachable for {selected_suction_name}; "
                "keeping the package at the high safe point and trying only physically "
                "verified D fallback angles. "
                f"Controller error: {aligned_d_exc}"
            )
            current_high_delta_deg = placement_delta_deg
            fallback_angles_deg = verified_placement_fallback_angles_deg(
                requested_placement_delta_deg,
                current_high_delta_deg,
                verified_placement_angles_for_cup(args, selected_suction_name),
            )
            if not args.allow_degraded_placement:
                fallback_angles_deg = [
                    angle
                    for angle in fallback_angles_deg
                    if placement_alignment_error_deg(
                        requested_placement_delta_deg, angle
                    ) <= float(args.max_placement_alignment_error_deg)
                ]
            for trial_delta_deg in fallback_angles_deg:
                trial_alignment_error_deg = placement_alignment_error_deg(
                    requested_placement_delta_deg, trial_delta_deg
                )
                print(
                    f"Trying verified placement rotation {trial_delta_deg:+.2f}deg "
                    f"after D rejected {current_high_delta_deg:+.2f}deg; "
                    f"predicted long-edge error={trial_alignment_error_deg:.2f}deg."
                )
                trial_safe_tcp, _trial_safe_center = rotate_selected_cup_at_functional_waypoint(
                    WAYPOINT_D_ROTATE_SAFE,
                    trial_delta_deg,
                    selected_suction_cup,
                    robot,
                    stop_options,
                    start_local_z_deg=current_high_delta_deg,
                )
                current_high_delta_deg = local_z_offset_between_waypoints_deg(
                    WAYPOINT_D_ROTATE_SAFE, trial_safe_tcp, selected_suction_cup
                )
                trial_d = rotate_waypoint_about_local_z(
                    WAYPOINT_D,
                    current_high_delta_deg,
                    name="D[reduced-reachable-search]",
                )
                trial_loaded_d, trial_center_d_mm = placement_tcp_waypoint_for_selected_cup(
                    trial_d, selected_suction_cup
                )
                try:
                    move_loaded_transfer_with_singularity_fallback(
                        trial_loaded_d, robot, stop_options
                    )
                except Exception as trial_d_exc:
                    if (is_operator_software_stop_error(trial_d_exc)
                            or is_robot_power_or_safety_state_error(trial_d_exc)):
                        raise
                    if not (is_no_ik_solution_error(trial_d_exc)
                            or is_path_singularity_error(trial_d_exc)):
                        raise
                    print(
                        f"D remains unreachable at verified angle "
                        f"{current_high_delta_deg:+.2f}deg; trying the next verified route. "
                        f"Controller error: {trial_d_exc}"
                    )
                    continue
                placement_delta_deg = current_high_delta_deg
                placed_local_z_deg = current_high_delta_deg
                aligned_waypoint_d = trial_d
                loaded_waypoint_d = trial_loaded_d
                selected_cup_center_d_mm = trial_center_d_mm
                reached_d = True
                print(
                    f"D-reachable verified placement rotation is "
                    f"{placed_local_z_deg:+.2f}deg; actual long-edge error="
                    f"{placement_alignment_error_deg(requested_placement_delta_deg, placed_local_z_deg):.2f}deg."
                )
                break

            if not reached_d:
                raise RuntimeError(
                    "No physically verified placement angle reached D within the configured "
                    "alignment policy. The package remains attached; use the verified recovery "
                    "route or operator handling instead of an unverified joint-branch change."
                )
        actual_alignment_error_deg = placement_alignment_error_deg(
            requested_placement_delta_deg, placed_local_z_deg
        )
        placement_quality = (
            "aligned"
            if actual_alignment_error_deg <= float(args.max_placement_alignment_error_deg)
            else "degraded"
        )
        if placement_quality == "degraded" and not args.allow_degraded_placement:
            raise RuntimeError(
                f"D was reached with long-edge error {actual_alignment_error_deg:.2f}deg, "
                f"above the allowed {args.max_placement_alignment_error_deg:.2f}deg; "
                "suction remains enabled."
            )
        print(
            f"At functional placement D for candidate #{candidate.index}; "
            f"{selected_suction_name} cup/package center="
            f"{selected_cup_center_d_mm.round(1).tolist()}; "
            f"placement_quality={placement_quality} requested_rotation="
            f"{requested_placement_delta_deg:+.2f}deg actual_rotation="
            f"{placed_local_z_deg:+.2f}deg alignment_error={actual_alignment_error_deg:.2f}deg; "
            f"disabling {selected_suction_name} suction "
            f"DO{args.suction_do_board}_{selected_suction_port}; "
            "lifting immediately to D-rotate-safe."
        )
        set_all_suction_outputs(robot, args, False)
        suction_active = False

        barcode_listener_threads: list[threading.Thread] = []

        def start_barcode_listener_at_safe_point() -> None:
            if args.disable_barcode_reader:
                start_top_front_barcode_listener(args, stop_options)
                return
            print(
                f"At D-rotate-safe; waiting {args.place_dwell_s:.1f}s before creating "
                "the top/front TCP barcode server."
            )
            wait_with_operator_stop(args.place_dwell_s, robot, stop_options)
            listener = start_top_front_barcode_listener(args, stop_options)
            if listener is not None:
                barcode_listener_threads.append(listener)

        return_empty_from_d_via_rotation_safe(
            selected_suction_cup,
            placed_local_z_deg,
            robot,
            stop_options,
            on_safe_arrival=start_barcode_listener_at_safe_point,
        )

        next_approach_plan: ApproachPlan | None = None
        if next_candidate is not None and next_candidate.motion_safe:
            print(
                f"Returning empty through C/B and continuing directly to "
                f"candidate #{next_candidate.index} A point."
            )
            move_waypoint(WAYPOINT_C, robot, approach_options)
            move_waypoint(WAYPOINT_B, robot, approach_options)

            # Replan from the actual B pose, then run the same dual-cup
            # selection used on a normal candidate turn. The old shortcut
            # prepositioned every next candidate with the primary cup and
            # bypassed secondary-cup scoring entirely.
            (
                next_approach_xyz_mm,
                next_pickup_xyz_mm,
                _next_target_rpy_deg,
                _next_rpy_mode,
                next_rpy_candidates,
            ) = compute_approach_geometry(next_candidate, robot, args)
            next_selection_origin_pose = robot.read_current_pose()
            next_suction_plans = build_suction_approach_plans(
                next_candidate,
                scene_candidates or [next_candidate],
                next_selection_origin_pose.translation_mm(),
                next_selection_origin_pose.rpy_deg_xyz(),
                next_approach_xyz_mm,
                next_pickup_xyz_mm,
                next_rpy_candidates,
                args,
                roi_config,
            )
            last_next_error: Exception | None = None
            for next_suction_plan in next_suction_plans:
                print(
                    "Selected next-candidate ranked plan for final 3-D cup-volume check: "
                    f"candidate=#{next_candidate.index} cup={next_suction_plan.cup.name}."
                )
                if not suction_plan_volume_clear(next_candidate, next_suction_plan, args):
                    print(
                        "Next-candidate ranked plan failed final 3-D cup-volume check; "
                        "checking the next ranked plan."
                    )
                    continue
                try:
                    print(
                        f"Trying next dual-suction approach: cup={next_suction_plan.cup.name} "
                        f"DO{args.suction_do_board}_{next_suction_plan.cup.do_port} "
                        f"RPY(deg)={next_suction_plan.rpy_deg.round(2).tolist()} "
                        f"score={next_suction_plan.score:.1f}"
                    )
                    next_commanded_rpy_deg = move_empty_approach_to_a(
                        robot,
                        next_suction_plan.approach_xyz_mm,
                        next_suction_plan.rpy_deg,
                        approach_options,
                    )
                    next_approach_plan = ApproachPlan(
                        candidate_index=next_candidate.index,
                        approach_xyz_mm=next_suction_plan.approach_xyz_mm,
                        pickup_xyz_mm=next_suction_plan.pickup_xyz_mm,
                        rpy_deg=next_commanded_rpy_deg,
                        rpy_mode=next_suction_plan.rpy_mode,
                        suction_name=next_suction_plan.cup.name,
                        suction_do_port=next_suction_plan.cup.do_port,
                    )
                    break
                except Exception as exc:
                    last_next_error = exc
                    if is_operator_software_stop_error(exc):
                        raise RuntimeError(
                            "Operator software stop is latched; aborting next-candidate "
                            "prepositioning and the complete batch."
                        ) from exc
                    if is_robot_power_or_safety_state_error(exc):
                        raise
                    print(
                        f"Next dual-suction approach with {next_suction_plan.cup.name} failed; "
                        "trying the next cup/pose plan. "
                        f"Controller error: {exc}"
                    )
            if next_approach_plan is None:
                print(
                    f"Next candidate #{next_candidate.index} was not prepositioned; "
                    "the batch will continue with a fresh approach on that candidate. "
                    f"Last controller error: {last_next_error}"
                )
        else:
            print(
                "Returning through C without a stop; "
                f"pass-through zone={pass_through_zone_mm:.1f} mm, B is the return endpoint."
            )
            # Suction is already off and the tool is empty here, so return via
            # the original TCP-based C/B clearance poses, not loaded C/D cup
            # compensation.
            move_waypoint(WAYPOINT_C, robot, approach_options)
            move_waypoint(WAYPOINT_B, robot, approach_options)
        for barcode_listener in barcode_listener_threads:
            barcode_listener.join(timeout=max(0.1, args.barcode_reader_timeout_s + 0.5))
    except Exception as exc:
        if waybill_inspector is not None:
            waybill_inspector.cancel_capture()
        if "suction_active" in locals() and suction_active and robot is not None:
            if args.suction_off_on_error:
                try:
                    set_suction_output(robot, args, False, do_port=selected_suction_port)
                    suction_active = False
                except Exception as suction_exc:
                    print(f"Warning: failed to turn suction off after batch error: {suction_exc}")
            else:
                print(
                    f"Warning: suction DO{args.suction_do_board}_{selected_suction_port} may still be ON; "
                    "the package may still be attached. Handle it manually before continuing."
                )
        print(
            f"Batch motion stopped at candidate #{candidate.index}: {exc}\n"
            "The remaining candidates were not executed."
        )
        return False, None, False

    print(f"Completed batch candidate #{candidate.index}.")
    return True, next_approach_plan, True


def execute_all_candidates(
    candidates: list[ClusterCandidate],
    robot: XCoreRobotClient | None,
    motion_options: MotionOptions,
    dry_run: bool,
    args: argparse.Namespace,
    waybill_inspector: AsyncWaybillInspector | None = None,
    roi_config: RoiConfig | None = None,
) -> BatchExecutionResult:
    print(
        f"Selecting one package from {len(candidates)} candidate(s). "
        "Enter/manual confirmation accepted; suction IOs="
        f"{'disabled' if args.disable_suction_io or dry_run or robot is None else ','.join(f'DO{args.suction_do_board}_{cup.do_port}' for cup in suction_cup_specs(args))}."
    )
    if robot is None or dry_run:
        for candidate in candidates:
            surface_normal = np.asarray(candidate.normal_base, dtype=np.float64)
            norm = float(np.linalg.norm(surface_normal))
            if norm < 1e-9:
                print(f"[DRY-RUN] Candidate #{candidate.index}: invalid normal.")
                continue
            surface_normal /= norm
            approach_xyz_mm = target_point_for_candidate(
                candidate.point_base_mm,
                surface_normal,
                args.standoff_mm,
                args.standoff_mode,
            )
            pickup_xyz_mm = approach_xyz_mm - args.pickup_down_mm * surface_normal
            print(
                f"[DRY-RUN] candidate #{candidate.index} {candidate.class_name}: "
                f"A={approach_xyz_mm.round(1).tolist()} "
                f"pickup={pickup_xyz_mm.round(1).tolist()} "
                f"A*={[WAYPOINT_A_STAR.x_mm, WAYPOINT_A_STAR.y_mm, WAYPOINT_A_STAR.z_mm]} "
                f"B={WAYPOINT_B.x_mm, WAYPOINT_B.y_mm, WAYPOINT_B.z_mm} "
                f"C={WAYPOINT_C.x_mm, WAYPOINT_C.y_mm, WAYPOINT_C.z_mm} "
                f"D={WAYPOINT_D.x_mm, WAYPOINT_D.y_mm, WAYPOINT_D.z_mm} "
                f"pass_through_zone={args.pass_through_zone_mm:.1f}mm "
                "forward_stop_points=C,D return_to_next_pass_through=C,B "
                f"suction_DOs={[f'DO{args.suction_do_board}_{cup.do_port}' for cup in suction_cup_specs(args)]} "
                f"secondary_offset_tool_y={args.secondary_suction_offset_y_mm:.1f}mm"
            )
        return BatchExecutionResult.COMPLETED

    motion_candidates = [candidate for candidate in candidates if candidate.motion_safe]
    skipped_count = len(candidates) - len(motion_candidates)
    if skipped_count:
        print(f"Skipping {skipped_count} blocked candidate(s); only motion-safe packages will be executed.")
    if not motion_candidates:
        print("No motion-safe candidate is available for robot motion.")
        return BatchExecutionResult.NO_SAFE_PLAN

    # The scene becomes stale as soon as one package is removed. Try ranked
    # candidates only until one succeeds, then force a fresh YOLO/depth pass.
    for candidate in motion_candidates:
        print(
            f"Next-pick choice #{candidate.index}: Z={candidate.point_base_mm[2]:.1f} mm "
            f"score={candidate.selection_score:.3f} confidence={candidate.confidence:.2f} "
            f"flatness={candidate.flatness_mm:.2f} mm ({candidate.selection_note})."
        )
        ok, _prepositioned_approach, completed = execute_candidate_sequence(
            candidate,
            robot,
            motion_options,
            args,
            next_candidate=None,
            prepositioned_approach=None,
            waybill_inspector=waybill_inspector,
            # Include blocked detections: they cannot be selected for motion,
            # but remain physical obstacles for the unused cup.
            scene_candidates=candidates,
            roi_config=roi_config,
        )
        if not ok:
            return BatchExecutionResult.MOTION_ERROR
        if completed:
            print("One package completed; forcing fresh YOLO/depth analysis before choosing another.")
            return BatchExecutionResult.COMPLETED
    else:
        print("No package completed successfully; automatic retry is disabled to avoid an endless loop.")
        return BatchExecutionResult.NO_SAFE_PLAN


def print_analysis_summary(analysis: AnalysisResult, args: argparse.Namespace) -> None:
    print(
        f"Analysis complete: {len(analysis.candidates)} candidate(s), "
        "sorted high to low."
    )
    for note in analysis.notes:
        print(f"  note: {note}")
    if analysis.debug_bgr is not None:
        debug_path = Path(args.debug_image)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), analysis.debug_bgr)
        print(f"  note: debug_image={debug_path}")
    for candidate in analysis.candidates:
        surface_tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(candidate.normal_base[2]), -1.0, 1.0)
                )
            )
        )
        print(
            f"  #{candidate.index}: region={candidate.support_region_name} "
            f"type={candidate.class_name} "
            f"cls={candidate.class_id} "
            f"conf={candidate.confidence:.2f} "
            f"source={candidate.detection_source} "
            f"height={candidate.height_mm:.1f} mm "
            f"flatness={candidate.flatness_mm:.2f} mm "
            f"points={candidate.point_count} "
            f"filter={candidate.filter_note} "
            f"surface XYZ(mm)={candidate.point_base_mm.round(1).tolist()} "
            f"normal(base)={candidate.normal_base.round(4).tolist()} "
            f"tilt={surface_tilt_deg:.1f}deg"
        )
        if candidate.short_axis_base is not None:
            print(
                f"      YOLO short edge (base)={candidate.short_axis_base.round(4).tolist()} "
                "(used to pre-orient pickup for long-edge placement at D)"
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.robot_speed_mm_s <= 0.0 or not np.isfinite(args.robot_speed_mm_s):
        parser.error("--robot-speed-mm-s must be a positive finite value")
    if args.robot_zone_mm < 0.0 or not np.isfinite(args.robot_zone_mm):
        parser.error("--robot-zone-mm must be a non-negative finite value")
    if args.robot_timeout_s <= 0.0 or not np.isfinite(args.robot_timeout_s):
        parser.error("--robot-timeout-s must be a positive finite value")
    if args.standoff_mm < 0.0 or not np.isfinite(args.standoff_mm):
        parser.error("--standoff-mm must be a non-negative finite value")
    if args.pickup_down_mm < 0.0 or not np.isfinite(args.pickup_down_mm):
        parser.error("--pickup-down-mm must be a non-negative finite value")
    if not args.verified_placement_angles_deg or not all(
        np.isfinite(angle) for angle in args.verified_placement_angles_deg
    ):
        parser.error("--verified-placement-angles-deg must contain finite angles")
    for option_name in (
        "primary_verified_placement_angles_deg",
        "secondary_verified_placement_angles_deg",
    ):
        configured_angles = getattr(args, option_name)
        if configured_angles is not None and (
            not configured_angles or not all(np.isfinite(angle) for angle in configured_angles)
        ):
            parser.error(f"--{option_name.replace('_', '-')} must contain finite angles")
    if (not np.isfinite(args.max_placement_alignment_error_deg)
            or not 0.0 <= args.max_placement_alignment_error_deg <= 90.0):
        parser.error("--max-placement-alignment-error-deg must be in [0, 90]")
    if args.placement_tcp_max_reach_mm <= 0.0 or np.isnan(args.placement_tcp_max_reach_mm):
        parser.error("--placement-tcp-max-reach-mm must be positive")
    if np.isnan(args.placement_tcp_y_max_mm):
        parser.error("--placement-tcp-y-max-mm must not be NaN")
    if args.placement_rotation_score_weight < 0.0 or not np.isfinite(
        args.placement_rotation_score_weight
    ):
        parser.error("--placement-rotation-score-weight must be a non-negative finite value")
    if args.place_dwell_s < 0.0 or not np.isfinite(args.place_dwell_s):
        parser.error("--place-dwell-s must be a non-negative finite value")
    if not 1 <= args.barcode_reader_port <= 65535:
        parser.error("--barcode-reader-port must be in the range 1..65535")
    if args.barcode_reader_timeout_s <= 0.0 or not np.isfinite(args.barcode_reader_timeout_s):
        parser.error("--barcode-reader-timeout-s must be a positive finite value")
    if not np.isfinite(args.secondary_suction_offset_y_mm):
        parser.error("--secondary-suction-offset-y-mm must be finite")
    if args.dual_suction_clearance_mm < 0.0 or not np.isfinite(args.dual_suction_clearance_mm):
        parser.error("--dual-suction-clearance-mm must be a non-negative finite value")
    if args.unused_cup_min_clearance_mm < 0.0 or not np.isfinite(args.unused_cup_min_clearance_mm):
        parser.error("--unused-cup-min-clearance-mm must be a non-negative finite value")
    if args.suction_cup_collision_radius_mm < 0.0 or not np.isfinite(args.suction_cup_collision_radius_mm):
        parser.error("--suction-cup-collision-radius-mm must be a non-negative finite value")
    if args.cup_volume_min_points < 1:
        parser.error("--cup-volume-min-points must be at least 1")
    if (not np.isfinite(args.top_plane_ransac_threshold_mm)
            or args.top_plane_ransac_threshold_mm <= 0.0):
        parser.error("--top-plane-ransac-threshold-mm must be a positive finite value")
    if args.top_plane_ransac_iterations < 1:
        parser.error("--top-plane-ransac-iterations must be at least 1")
    if (not np.isfinite(args.cup_contact_surface_match_mm)
            or args.cup_contact_surface_match_mm < 0.0):
        parser.error("--cup-contact-surface-match-mm must be a non-negative finite value")
    if args.fixed_x_fallback_yaw_step_deg <= 0.0 or not np.isfinite(args.fixed_x_fallback_yaw_step_deg):
        parser.error("--fixed-x-fallback-yaw-step-deg must be a positive finite value")
    if args.fixed_x_fallback_yaw_max_deg < 0.0 or not np.isfinite(args.fixed_x_fallback_yaw_max_deg):
        parser.error("--fixed-x-fallback-yaw-max-deg must be a non-negative finite value")
    if args.waybill_c_settle_s < 0.0 or not np.isfinite(args.waybill_c_settle_s):
        parser.error("--waybill-c-settle-s must be a non-negative finite value")
    if args.waybill_post_c_capture_s <= 0.0 or not np.isfinite(args.waybill_post_c_capture_s):
        parser.error("--waybill-post-c-capture-s must be a positive finite value")
    if args.waybill_c_dwell_s < 0.0 or not np.isfinite(args.waybill_c_dwell_s):
        parser.error("--waybill-c-dwell-s must be a non-negative finite value")
    if args.waybill_result_timeout_s <= 0.0 or not np.isfinite(args.waybill_result_timeout_s):
        parser.error("--waybill-result-timeout-s must be a positive finite value")
    if args.waybill_capture_duration_s <= args.waybill_c_settle_s + args.waybill_post_c_capture_s:
        parser.error("--waybill-capture-duration-s must exceed C settle plus static capture time")
    if not args.disable_secondary_suction and args.secondary_suction_do_port == args.suction_do_port:
        parser.error("primary and secondary suction cups must use different DO ports")
    if args.force_suction_cups:
        forced = set(args.force_suction_cups)
        disabled_forced = []
        if "2" in forced and args.disable_secondary_suction:
            disabled_forced.append("2")
        if "3" in forced and args.disable_third_suction:
            disabled_forced.append("3")
        if "4" in forced and args.disable_fourth_suction:
            disabled_forced.append("4")
        if disabled_forced:
            parser.error(
                "forced suction cup(s) are also disabled: " + ", ".join(disabled_forced)
            )
    if not args.dry_run and not args.acknowledge_verified_tcp:
        parser.error(
            "real robot motion is locked after the reported tool collision; physically verify "
            f"the {args.tool_name or 'active'} TCP and all fixed poses at low speed, then pass "
            "--acknowledge-verified-tcp"
        )
    if args.workspace_x_min_mm >= args.workspace_x_max_mm:
        parser.error("workspace X minimum must be smaller than its maximum")
    if args.workspace_y_min_mm >= args.workspace_y_max_mm:
        parser.error("workspace Y minimum must be smaller than its maximum")
    if args.workspace_z_min_mm >= args.workspace_z_max_mm:
        parser.error("workspace Z minimum must be smaller than its maximum")
    if (args.tcp_override_xyz_mm is None) != (args.tcp_override_rpy_deg is None):
        parser.error("--tcp-override-xyz-mm and --tcp-override-rpy-deg must be provided together")
    if args.rpy_mode == "fixed" and args.fixed_pickup_rpy_deg is None:
        parser.error("--rpy-mode fixed requires --fixed-pickup-rpy-deg RX RY RZ")
    if args.ignore_workspace_filter and not args.dry_run:
        parser.error("--ignore-workspace-filter 只能和 --dry-run 一起使用，避免真实运动绕过工作空间安全限制。")

    camera_matrix = load_camera_matrix(args.intrinsics)
    camera_point_to_base = load_camera_point_to_base_transform(args.hand_eye)
    yolo_model = None
    if args.detector == "yolo":
        print(f"Loading YOLO model: {args.yolo_model}")
        yolo_model = load_yolo_detector(args.yolo_model)
        print(f"YOLO loaded: task={getattr(yolo_model, 'task', 'unknown')}, names={getattr(yolo_model, 'names', {})}")

    waybill_inspector: AsyncWaybillInspector | None = None
    if args.enable_waybill_inspection:
        if not args.waybill_camera_password:
            parser.error(
                "Waybill inspection requires a camera password. "
                "Set HIKVISION_PASSWORD or pass --waybill-camera-password."
            )
        print(f"Loading waybill model: {args.waybill_model}")
        waybill_inspector = AsyncWaybillInspector(
            camera_ip=args.waybill_camera_ip,
            username=args.waybill_camera_username,
            password=args.waybill_camera_password,
            model_path=args.waybill_model,
            barcode_model_path=args.barcode_model,
            output_dir=args.waybill_output_dir,
            confidence=args.waybill_conf,
            barcode_confidence=args.barcode_conf,
            capture_count=args.waybill_capture_count,
            capture_interval_s=args.waybill_capture_interval_s,
            capture_duration_s=args.waybill_capture_duration_s,
            c_settle_s=args.waybill_c_settle_s,
            post_c_capture_s=args.waybill_post_c_capture_s,
            request_timeout_s=args.waybill_request_timeout_s,
        )
        print(
            f"Waybill inspection ready: camera={args.waybill_camera_ip}, "
            f"max_clip={args.waybill_capture_duration_s:.2f}s, "
            f"C_settle={args.waybill_c_settle_s:.2f}s, "
            f"post_C={args.waybill_post_c_capture_s:.2f}s, "
            f"C_dwell={args.waybill_c_dwell_s:.2f}s, "
            f"max_frames={args.waybill_capture_count}, interval={args.waybill_capture_interval_s:.2f}s"
        )

    camera = OrbbecRGBDCamera(
        CameraOpenOptions(
            width=args.width,
            height=args.height,
            fps=args.fps,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            depth_fps=args.depth_fps,
            align_mode=args.align_mode,
            wait_timeout_ms=args.wait_timeout_ms,
        )
    )
    camera.start()

    robot: XCoreRobotClient | None = None
    motion_options = MotionOptions(
        motion="movej",
        speed_mm_s=args.robot_speed_mm_s,
        zone_mm=args.robot_zone_mm,
        timeout_s=args.robot_timeout_s,
        use_current_conf_data=args.use_current_conf_data,
    )
    print(f"Robot motion settings: speed={args.robot_speed_mm_s:.1f} mm/s zone={args.robot_zone_mm:.1f} mm")
    if args.force_suction_cups:
        selected_cups = selected_suction_cup_specs(args)
        print(
            "Forced pickup cups: "
            + ", ".join(
                f"{cup.name}=DO{args.suction_do_board}_{cup.do_port}" for cup in selected_cups
            )
            + "; all configured cup outputs are still cleared before each pickup."
        )
    operator_stop_requested = False
    operator_stop_lock = threading.Lock()

    if not args.dry_run:
        try:
            robot = XCoreRobotClient(args.robot_ip)
            robot.prepare_motion(args.robot_speed_mm_s, args.robot_zone_mm)
            if args.tool_name:
                if args.tcp_override_xyz_mm is not None:
                    toolset = robot.set_toolset_with_tcp_override(
                        args.tool_name,
                        args.wobj_name,
                        args.tcp_override_xyz_mm,
                        args.tcp_override_rpy_deg,
                    )
                    print(
                        f"Robot toolset selected: tool={args.tool_name} wobj={args.wobj_name}; "
                        "session-only TCP override active (controller record unchanged)"
                    )
                else:
                    toolset = robot.set_toolset_by_name(args.tool_name, args.wobj_name)
                    print(f"Robot toolset selected: tool={args.tool_name} wobj={args.wobj_name}")
            else:
                toolset = robot.read_toolset()
                print("Robot toolset selected: current controller/SDK toolset")
            print(
                "Active TCP relative to flange: "
                f"XYZ(mm)={toolset.end_translation_mm().round(3).tolist()} "
                f"RPY(deg)={toolset.end_rpy_deg_xyz().round(3).tolist()}"
            )
            if args.disable_suction_io:
                print(
                    "Suction IO control disabled; outputs "
                    f"{[f'DO{args.suction_do_board}_{cup.do_port}' for cup in suction_cup_specs(args)]} "
                    "will not be changed."
                )
            else:
                set_all_suction_outputs(robot, args, False)
            print(f"Robot connected: {args.robot_ip}")
        except Exception as exc:
            robot = None
            camera.stop()
            if waybill_inspector is not None:
                waybill_inspector.close()
            print(
                "Robot connection/setup failed; refusing to silently continue in dry-run mode. "
                f"Use --dry-run explicitly for simulation. Error: {exc}"
            )
            return 1

    latest_color = None
    latest_depth = None
    latest_depth_display = None
    analysis: AnalysisResult | None = None
    selected_index: int | None = None
    initial_analysis_done = False
    next_analysis_time = 0.0
    quit_requested = False
    batch_auto_enabled = False
    analysis_id = 0
    consumed_analysis_id = 0
    next_auto_allowed_time = 0.0
    frames_to_discard = 0
    roi_config = load_roi_config(args.roi_json)
    roi_edit_target: str | None = None
    roi_draft_points: list[tuple[int, int]] = []

    click_state = {"pending": None}

    def mouse_callback(event, x, y, _flags, _param):
        nonlocal roi_edit_target, roi_draft_points
        if roi_edit_target is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_draft_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and roi_draft_points:
                roi_draft_points.pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["pending"] = (x, y)

    if not args.headless:
        cv2.namedWindow(COLOR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(COLOR_WINDOW, mouse_callback)
        cv2.setMouseCallback(DEPTH_WINDOW, mouse_callback)

    def request_operator_stop(reason: str = "SPACE") -> None:
        nonlocal operator_stop_requested, batch_auto_enabled
        with operator_stop_lock:
            first_request = not operator_stop_requested
            operator_stop_requested = True
            batch_auto_enabled = False
        if not first_request:
            return
        print(
            f"{reason} pressed: stopping robot motion, clearing queued commands, "
            "and aborting continuous batch mode. Suction outputs are intentionally kept unchanged."
        )
        if robot is not None:
            try:
                robot.stop_motion()
            except Exception as exc:
                print(f"Warning: immediate robot software stop failed: {exc}")

    def request_quit() -> None:
        nonlocal quit_requested
        quit_requested = True
        request_operator_stop("Q")

    remote_keys: queue.Queue[int] = queue.Queue()

    def stdin_control_loop() -> None:
        command_keys = {
            "ANALYZE": ord("d"),
            "START": ord("a"),
            "RESET": ord("r"),
        }
        for raw_line in sys.stdin:
            command = raw_line.strip().upper()
            if not command:
                continue
            if command == "STOP":
                request_operator_stop("GUI STOP")
            elif command == "QUIT":
                request_quit()
            elif command in command_keys:
                remote_keys.put(command_keys[command])
            else:
                print(f"Unknown GUI control command ignored: {command}")

    if args.control_stdin:
        threading.Thread(
            target=stdin_control_loop,
            name="gui-stdin-control",
            daemon=True,
        ).start()
        print("GUI stdin control enabled: ANALYZE, START, RESET, STOP, QUIT.")

    def merge_remote_key(local_key: int) -> int:
        try:
            return remote_keys.get_nowait()
        except queue.Empty:
            return local_key

    gui_frame_dir = Path(args.gui_frame_dir).resolve() if args.gui_frame_dir else None
    if gui_frame_dir is not None:
        gui_frame_dir.mkdir(parents=True, exist_ok=True)
    last_gui_frame_time = 0.0

    def publish_gui_frames(color_image: np.ndarray, depth_image: np.ndarray) -> None:
        nonlocal last_gui_frame_time
        if gui_frame_dir is None:
            return
        now = time.monotonic()
        if now - last_gui_frame_time < max(0.05, float(args.gui_frame_interval_s)):
            return
        last_gui_frame_time = now
        for filename, image in (("color.jpg", color_image), ("depth.jpg", depth_image)):
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            target = gui_frame_dir / filename
            temporary = gui_frame_dir / f"{filename}.tmp"
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, target)

    def poll_operator_stop_key() -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            request_operator_stop("SPACE")
        elif key == ord("q"):
            request_quit()
        return operator_stop_requested

    motion_options.stop_requested = poll_operator_stop_key
    space_stop_monitor = GlobalSpaceStopMonitor(lambda: request_operator_stop("GLOBAL SPACE"))
    if space_stop_monitor.start():
        print(
            "Global SPACE software stop enabled: press Space at any time to stop motion and "
            "clear queued commands (suction remains unchanged)."
        )
    else:
        print("SPACE software stop is available only while an OpenCV window has keyboard focus.")

    def schedule_next_analysis() -> None:
        nonlocal next_analysis_time
        if args.analysis_refresh_s > 0.0:
            interval_s = max(0.1, float(args.analysis_refresh_s))
        elif analysis is None:
            interval_s = 2.0
        else:
            interval_s = float("inf")
        next_analysis_time = time.monotonic() + interval_s

    def run_batch_once(trigger: str) -> bool:
        nonlocal analysis, selected_index, next_analysis_time, consumed_analysis_id
        nonlocal next_auto_allowed_time, frames_to_discard, latest_color, latest_depth, latest_depth_display
        nonlocal batch_auto_enabled
        if analysis is None:
            return False
        if analysis_id <= consumed_analysis_id:
            print(
                f"{trigger}: analysis #{analysis_id} was already consumed; "
                "waiting for a fresh camera analysis."
            )
            schedule_next_analysis()
            return False
        motion_safe_count = sum(1 for candidate in analysis.candidates if candidate.motion_safe)
        if motion_safe_count <= 0:
            consumed_analysis_id = analysis_id
            print(f"{trigger}: analysis #{analysis_id} has no motion-safe package; waiting for the next refresh.")
            schedule_next_analysis()
            return False
        print(
            f"{trigger}: selecting one package from {motion_safe_count} motion-safe candidate(s) "
            f"in analysis #{analysis_id}."
        )
        batch_result = execute_all_candidates(
            analysis.candidates,
            robot,
            motion_options,
            args.dry_run or robot is None,
            args,
            waybill_inspector=waybill_inspector,
            roi_config=roi_config,
        )
        if batch_result == BatchExecutionResult.MOTION_ERROR:
            batch_auto_enabled = False
            print(
                "Continuous batch mode disabled after a motion error. "
                "Resolve the robot/suction state and press Enter to start again."
            )
        elif batch_result == BatchExecutionResult.NO_SAFE_PLAN:
            batch_auto_enabled = False
            print(
                "Continuous batch mode paused: this analysis has no plan that passed "
                "the geometric safety checks. The robot controller did not report a motion error. "
                "Press Enter to analyze and try again."
            )
        consumed_analysis_id = analysis_id
        analysis = None
        selected_index = None
        latest_color = None
        latest_depth = None
        latest_depth_display = None
        frames_to_discard = max(0, int(args.post_batch_discard_frames))
        next_auto_allowed_time = time.monotonic() + max(0.0, float(args.auto_cycle_settle_s))
        next_analysis_time = next_auto_allowed_time
        print(
            f"{trigger}: consumed analysis #{consumed_analysis_id}; "
            f"discarding {frames_to_discard} camera frame(s), "
            f"next auto analysis allowed after {args.auto_cycle_settle_s:.1f}s."
        )
        return batch_result == BatchExecutionResult.COMPLETED

    try:
        while True:
            analysis_updated = False
            frames = camera.get_frames(args.wait_timeout_ms)
            if frames is not None:
                if frames_to_discard > 0:
                    frames_to_discard -= 1
                    print(f"Discarding queued camera frame after batch; remaining={frames_to_discard}.")
                else:
                    latest_color, latest_depth, latest_depth_display = frames

            if latest_color is None or latest_depth is None or latest_depth_display is None:
                key = merge_remote_key(cv2.waitKey(1) & 0xFF)
                if key == ord("q"):
                    request_quit()
                    break
                if quit_requested:
                    break
                continue

            if roi_edit_target is None:
                now = time.monotonic()
                auto_refresh_enabled = args.analysis_refresh_s > 0.0
                refresh_due = (
                    now >= next_analysis_time
                    and now >= next_auto_allowed_time
                    and frames_to_discard <= 0
                    and (analysis is None or not initial_analysis_done or auto_refresh_enabled)
                )
                if refresh_due:
                    try:
                        analysis = analyze_scene(
                            latest_color.copy(),
                            latest_depth.copy(),
                            latest_depth_display.copy(),
                            camera_matrix,
                            camera_point_to_base,
                            args,
                            roi_config,
                            yolo_model,
                        )
                        analysis_id += 1
                        analysis_updated = True
                        selected_index = analysis.candidates[0].index if analysis.candidates else None
                        if not initial_analysis_done:
                            print_analysis_summary(analysis, args)
                            initial_analysis_done = True
                        else:
                            class_counts: dict[str, int] = {}
                            for item in analysis.candidates:
                                class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                            classes = ", ".join(
                                f"{name}:{count}" for name, count in sorted(class_counts.items())
                            )
                            print(
                                f"Analysis refreshed #{analysis_id}: {len(analysis.candidates)} candidate(s)"
                                + (f" [{classes}]" if classes else "")
                            )
                    except Exception as exc:
                        if not initial_analysis_done:
                            initial_analysis_done = True
                            print(f"Initial analysis failed: {exc}")
                        else:
                            print(f"Analysis refresh failed, keeping the previous result: {exc}")
                    finally:
                        schedule_next_analysis()

            if analysis is None or roi_edit_target is not None:
                color_canvas, depth_canvas = draw_live_overlay(
                    latest_color,
                    latest_depth_display,
                    roi_config,
                    roi_draft_points,
                    roi_edit_target,
                )
            else:
                color_canvas, depth_canvas = draw_analysis_overlay(
                    analysis,
                    selected_index,
                    roi_config,
                    roi_draft_points,
                    roi_edit_target,
                )

                pending = click_state["pending"]
                if pending is not None and roi_edit_target is None:
                    click_state["pending"] = None
                    candidate = pick_candidate(pending[0], pending[1], analysis.candidates)
                    if candidate is not None:
                        selected_index = candidate.index
                        print(
                            f"Selected candidate #{candidate.index}: "
                            f"type={candidate.class_name} "
                            f"cls={candidate.class_id} "
                            f"conf={candidate.confidence:.2f} "
                            f"region={candidate.support_region_name} "
                            f"source={candidate.detection_source} "
                            f"height={candidate.height_mm:.1f} mm "
                            f"score={candidate.selection_score:.3f} "
                            f"flatness={candidate.flatness_mm:.2f} mm "
                            f"filter={candidate.filter_note} "
                            f"surface XYZ(mm)={candidate.point_base_mm.round(1).tolist()}"
                        )
                        point_cloud_view = save_candidate_point_cloud_debug(candidate, args)
                        if point_cloud_view is not None:
                            cv2.namedWindow(POINT_CLOUD_WINDOW, cv2.WINDOW_NORMAL)
                            cv2.imshow(POINT_CLOUD_WINDOW, point_cloud_view)
                        execute_candidate(
                            candidate,
                            robot,
                            motion_options,
                            args.dry_run or robot is None,
                            args,
                        )
                    else:
                        nearest, distance_px = nearest_candidate_distance(
                            pending[0],
                            pending[1],
                            analysis.candidates,
                        )
                        if nearest is None:
                            print(f"No candidate was found near click ({pending[0]}, {pending[1]}).")
                        else:
                            print(
                                f"No candidate was found near click ({pending[0]}, {pending[1]}). "
                                f"Nearest is #{nearest.index}, distance={distance_px:.1f} px."
                            )

            publish_gui_frames(color_canvas, depth_canvas)
            if not args.headless:
                cv2.imshow(COLOR_WINDOW, color_canvas)
                cv2.imshow(DEPTH_WINDOW, depth_canvas)

            key = merge_remote_key(cv2.waitKey(1) & 0xFF)
            if quit_requested:
                break
            if key == ord("q"):
                request_quit()
                break
            if key == 32:
                request_operator_stop("SPACE")
                continue
            if key in (ord("m"), ord("b"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                if key == ord("m"):
                    roi_edit_target = "overall"
                elif key == ord("b"):
                    roi_edit_target = "exclude"
                elif key == ord("5"):
                    roi_edit_target = "suction_left"
                elif key == ord("6"):
                    roi_edit_target = "suction_right"
                else:
                    spec_index = int(chr(key)) - 1
                    roi_edit_target = SUPPORT_REGION_SPECS[spec_index].region_id
                roi_draft_points = []
                analysis = None
                selected_index = None
                next_analysis_time = 0.0
                click_state["pending"] = None
                if roi_edit_target == "overall":
                    print("Overall ROI edit mode: left click add points, right click undo, c/Enter save.")
                elif roi_edit_target == "exclude":
                    print("Exclude ROI edit mode: draw fixed obstacles, e.g. baffles. Left click add points, c/Enter save.")
                elif roi_edit_target in {"suction_left", "suction_right"}:
                    zone_name = roi_edit_target.removeprefix("suction_")
                    allowed = getattr(args, f"{zone_name}_zone_suction_cups")
                    print(
                        f"{zone_name.title()} suction ROI edit mode: candidate centers here "
                        f"are restricted to cup(s) {list(allowed)}."
                    )
                else:
                    spec = get_support_region_spec(roi_edit_target)
                    print(f"{spec.name} ROI edit mode: left click add points, right click undo, c/Enter save.")
                continue
            if key in (ord("c"), 13):
                if roi_edit_target is not None:
                    if len(roi_draft_points) < 3:
                        print("ROI needs at least 3 points.")
                    else:
                        polygon = np.asarray(roi_draft_points, dtype=np.int32)
                        if roi_edit_target == "overall":
                            roi_config.overall_polygon = polygon
                        elif roi_edit_target == "exclude":
                            roi_config.exclude_polygons.append(polygon)
                        elif roi_edit_target in {"suction_left", "suction_right"}:
                            zone_name = roi_edit_target.removeprefix("suction_")
                            roi_config.suction_zone_polygons[zone_name] = polygon
                        else:
                            roi_config.support_polygons[roi_edit_target] = polygon
                        save_roi_config(args.roi_json, roi_config)
                        roi_edit_target = None
                        roi_draft_points = []
                        analysis = None
                        selected_index = None
                        next_analysis_time = 0.0
                        print(f"ROI saved: {args.roi_json}")
                    continue
            if key == 27:
                if roi_edit_target is not None:
                    roi_edit_target = None
                    roi_draft_points = []
                    next_analysis_time = 0.0
                    print("ROI edit cancelled.")
                continue
            if key == ord("x"):
                if roi_edit_target == "overall":
                    roi_config.overall_polygon = None
                    print("Overall ROI cleared.")
                elif roi_edit_target == "exclude":
                    roi_config.exclude_polygons = []
                    print("Exclude ROIs cleared.")
                elif roi_edit_target in {"suction_left", "suction_right"}:
                    zone_name = roi_edit_target.removeprefix("suction_")
                    roi_config.suction_zone_polygons.pop(zone_name, None)
                    print(f"{zone_name.title()} suction ROI cleared.")
                elif roi_edit_target is not None:
                    roi_config.support_polygons.pop(roi_edit_target, None)
                    spec = get_support_region_spec(roi_edit_target)
                    print(f"{spec.name} ROI cleared.")
                else:
                    roi_config = RoiConfig(overall_polygon=None, support_polygons={}, exclude_polygons=[])
                    print("All ROI regions cleared.")
                roi_draft_points = []
                roi_edit_target = None
                analysis = None
                selected_index = None
                next_analysis_time = 0.0
                click_state["pending"] = None
                if (
                    roi_config.overall_polygon is None
                    and not roi_config.support_polygons
                    and not roi_config.exclude_polygons
                    and not roi_config.suction_zone_polygons
                ):
                    delete_roi_config(args.roi_json)
                else:
                    save_roi_config(args.roi_json, roi_config)
                continue
            if key == ord("d"):
                if roi_edit_target is not None:
                    print("Finish ROI editing first with c/Enter, or press Esc to cancel.")
                    continue
                if latest_color is None or latest_depth is None or latest_depth_display is None:
                    print("No frame is available yet.")
                    continue
                try:
                    analysis = analyze_scene(
                        latest_color.copy(),
                        latest_depth.copy(),
                        latest_depth_display.copy(),
                        camera_matrix,
                        camera_point_to_base,
                        args,
                        roi_config,
                        yolo_model,
                    )
                    analysis_id += 1
                    analysis_updated = True
                    selected_index = analysis.candidates[0].index if analysis.candidates else None
                    print_analysis_summary(analysis, args)
                    schedule_next_analysis()
                except Exception as exc:
                    analysis = None
                    selected_index = None
                    schedule_next_analysis()
                    print(f"Analysis failed: {exc}")
            if key == ord("r"):
                analysis = None
                selected_index = None
                next_analysis_time = 0.0
                click_state["pending"] = None
            if key in (13, ord("a")) and analysis is not None:
                if roi_edit_target is not None:
                    print("Finish ROI editing first with c/Enter, or press Esc to cancel.")
                    continue
                with operator_stop_lock:
                    operator_stop_requested = False
                if not batch_auto_enabled:
                    batch_auto_enabled = True
                    print("Continuous batch mode enabled: future refreshed detections will run automatically.")
                run_batch_once("Operator start")
                if quit_requested:
                    break
                continue
            if (
                batch_auto_enabled
                and analysis_updated
                and analysis is not None
                and roi_edit_target is None
                and not operator_stop_requested
            ):
                run_batch_once("Auto cycle")
                if quit_requested:
                    break
    finally:
        space_stop_monitor.stop()
        cv2.destroyAllWindows()
        camera.stop()
        if waybill_inspector is not None:
            waybill_inspector.close()
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
