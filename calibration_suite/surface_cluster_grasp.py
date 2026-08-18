from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree

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
from project0714_grasp.waybill_inspection import AsyncWaybillInspector


COLOR_WINDOW = "Project0714 Surface Grasp Color"
DEPTH_WINDOW = "Project0714 Surface Grasp Depth"
POINT_CLOUD_WINDOW = "Project0714 Candidate Point Cloud"

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


@dataclass
class SuctionApproachPlan:
    cup: SuctionCupSpec
    approach_xyz_mm: np.ndarray
    pickup_xyz_mm: np.ndarray
    rpy_deg: np.ndarray
    rpy_mode: str
    score: float
    travel_mm: float
    radial_reach_mm: float
    unused_cup_clearance_mm: float


# Fixed safe path waypoints, expressed in the active wobj0/tool coordinate setup.
# A* is the newly taught lift-and-clear transition point used on the loaded return path.
WAYPOINT_A_STAR = RobotWaypoint("A*", 110.628, -829.682, 482.697, -1.917, 2.748, -94.661)
WAYPOINT_B = RobotWaypoint("B", 779.155, -218.594, 487.419, -0.296, 1.428, -6.094)
WAYPOINT_C = RobotWaypoint("C", 916.271, 119.481, 326.308, -1.990, 5.492, -3.082)
WAYPOINT_D = RobotWaypoint("D", 452.368, 808.428, 284.504, -3.030, 0.839, 79.567)
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
        default=str(Path(__file__).resolve().parents[1] / "weights" / "waybill_yolo11n_best.pt"),
        help="YOLO model used to detect a waybill on the package bottom",
    )
    parser.add_argument("--waybill-conf", type=float, default=0.5, help="Waybill YOLO confidence threshold")
    parser.add_argument("--waybill-capture-count", type=int, default=15, help="Maximum settled C-point frames retained for inspection")
    parser.add_argument("--waybill-capture-interval-s", type=float, default=0.1, help="Background snapshot interval")
    parser.add_argument("--waybill-start-delay-s", type=float, default=0.5, help="Delay after waypoint B before bottom-camera capture starts")
    parser.add_argument("--waybill-capture-duration-s", type=float, default=30.0, help="Safety timeout from B prewarm through completion of C capture")
    parser.add_argument("--waybill-c-settle-s", type=float, default=0.6, help="Discard frames for this long after reaching C to let motion/exposure settle")
    parser.add_argument("--waybill-post-c-capture-s", type=float, default=2.5, help="Capture settled stationary frames for this long at C")
    parser.add_argument("--waybill-c-dwell-s", type=float, default=3.2, help="Minimum hold at C; automatically extended to cover settle plus capture")
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
    parser.add_argument("--rgb-nms-overlap", type=float, default=0.65, help="Overlap-over-smaller threshold for removing duplicate RGB rectangles")
    parser.add_argument("--min-rect-foreground-ratio", type=float, default=0.22, help="Minimum foreground ratio inside an RGB rectangle")
    parser.add_argument("--min-rect-foreground-pixels", type=int, default=250, help="Minimum foreground pixels inside an RGB rectangle")
    parser.add_argument("--max-rect-foreground-pixels", type=int, default=45000, help="Maximum foreground pixels inside one package rectangle")
    parser.add_argument("--min-candidate-height-mm", type=float, default=30.0, help="Minimum height above support plane for a valid package candidate")
    parser.add_argument("--final-candidate-nms-overlap", type=float, default=0.55, help="Final overlap threshold for removing duplicate package candidates")
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
        default="auto",
        help="Normal used for grasp orientation and standoff",
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
        help="Dwell time after disabling suction at waypoint D",
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
        default=5,
        help="xCore DO port number for the suction valve output, e.g. DO3_5 uses port 5",
    )
    parser.add_argument(
        "--secondary-suction-do-port",
        type=int,
        default=6,
        help="xCore DO port for the second suction cup; the default is DO3_6",
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
        help="Use only the primary TCP-centered suction cup",
    )
    parser.add_argument(
        "--dual-suction-clearance-mm",
        type=float,
        default=120.0,
        help="Preferred horizontal clearance between the unused cup and another detected package",
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
    return RoiConfig(
        overall_polygon=overall_polygon,
        support_polygons=support_polygons,
        exclude_polygons=exclude_polygons,
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


def prune_duplicate_candidates(
    candidates: list[ClusterCandidate],
    image_shape: tuple[int, int],
    overlap_threshold: float,
) -> list[ClusterCandidate]:
    def score(candidate: ClusterCandidate) -> tuple[float, float, float, float]:
        return (
            1.0 if candidate.motion_safe else 0.0,
            candidate.foreground_ratio,
            candidate.height_mm,
            -candidate.rect_area_px,
        )

    kept: list[ClusterCandidate] = []
    for candidate in sorted(candidates, key=score, reverse=True):
        if any(candidate_overlap_over_smaller(candidate, existing, image_shape) >= overlap_threshold for existing in kept):
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
) -> tuple[list[ClusterCandidate], dict[int, tuple[bool, float, int | None]]]:
    coverage: dict[int, tuple[bool, float, int | None]] = {}
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

    ranked = sorted(
        candidates,
        key=lambda item: (
            1 if coverage[id(item)][0] else 0,
            -float(item.point_base_mm[2]),
            -float(item.confidence),
            float(item.flatness_mm),
            -int(item.point_count),
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

    kept: list[RgbRectRegion] = []
    for region in sorted(regions, key=lambda item: item.contour_area_px, reverse=True):
        if any(rectangle_overlap_over_smaller(region, existing) >= args.rgb_nms_overlap for existing in kept):
            continue
        kept.append(region)

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


def target_rpy_candidates_for_candidate(
    candidate: ClusterCandidate,
    current_pose,
    args: argparse.Namespace,
    primary_rpy_deg: np.ndarray,
    primary_rpy_mode: str,
) -> list[tuple[str, np.ndarray]]:
    if not (args.align_normal_rpy or args.rpy_mode == "align-normal"):
        return [(primary_rpy_mode, primary_rpy_deg)]
    if primary_rpy_mode == "align_normal_fixed_x":
        fixed_rpy_deg = np.asarray(primary_rpy_deg, dtype=np.float64)
        candidates = [(primary_rpy_mode, fixed_rpy_deg)]
        if args.disable_fixed_x_yaw_fallback:
            return candidates

        step_deg = max(1.0, float(args.fixed_x_fallback_yaw_step_deg))
        max_deg = max(0.0, float(args.fixed_x_fallback_yaw_max_deg))
        offset_deg = step_deg
        while offset_deg <= max_deg + 1e-6:
            fallback_rpy_deg = compute_grasp_rpy_from_normal_and_fixed_x(
                candidate.normal_base,
                FIXED_GRASP_X_YAW_DEG + offset_deg,
                args.tool_contact_axis,
            )
            candidates.append(
                (
                    f"align_normal_fixed_x_fallback_yaw_{offset_deg:+.1f}deg",
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
    """Reject a numerically discontinuous command before it can wind the wrist."""
    previous = np.asarray(previous_rpy_deg, dtype=np.float64)
    target = np.asarray(target_rpy_deg, dtype=np.float64)
    delta = np.abs(target - previous)
    if float(np.max(delta)) > max_component_step_deg:
        raise RuntimeError(
            f"Unsafe RPY step for {label}: previous={previous.round(2).tolist()} "
            f"target={target.round(2).tolist()} delta={delta.round(2).tolist()} deg "
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

        top_centroid, top_normal = fit_plane_svd(top_points)
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

        top_centroid, top_normal = fit_plane_svd(rect_points)
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
            short_axis_camera=short_axis_camera,
            short_axis_base=short_axis_base,
        )
        loose_candidates.append(candidate)

    candidates = prune_duplicate_candidates(
        loose_candidates,
        color_bgr.shape[:2],
        args.final_candidate_nms_overlap,
    )
    candidates.sort(key=lambda item: (-item.point_base_mm[2], item.flatness_mm, -item.point_count))
    for idx, candidate in enumerate(candidates, start=1):
        candidate.index = idx
    candidates, coverage = rank_candidates_for_next_pick(candidates, color_bgr.shape[:2])
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
        f"height_sort=base_z",
        "pick_priority=uncovered_then_height_then_confidence_then_flatness",
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

        top_centroid, top_normal = fit_plane_svd(top_points)
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
        )
        loose_candidates.append(candidate)

        if motion_safe:
            strict_candidates.append(candidate)

    pruned_candidates = prune_duplicate_candidates(
        loose_candidates,
        color_bgr.shape[:2],
        args.final_candidate_nms_overlap,
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
    return AnalysisResult(color_bgr, depth_mm, depth_display, fallback_plane, candidates, notes, debug_bgr)


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
        "Enter: run batch sequence, Space: software stop, a: alias, r/d: refresh now, b: exclude ROI, m/1/2/3/4: edit ROI, q: quit",
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
            "auto-recognition idle; d/r: refresh now, b: add exclude ROI, m: overall ROI, 1/2/3/4: support ROIs, x: clear all, q: quit",
            f"overall_roi={'on' if roi_config.overall_polygon is not None else 'off'}",
            f"exclude_rois={len(roi_config.exclude_polygons)}",
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
    rpy_deg = (
        np.asarray(override_rpy_deg, dtype=np.float64)
        if override_rpy_deg is not None
        else np.asarray([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg], dtype=np.float64)
    )
    print(
        f"Moving waypoint {waypoint.name}: "
        f"XYZ(mm)={[waypoint.x_mm, waypoint.y_mm, waypoint.z_mm]} "
        f"RPY(deg)={rpy_deg.round(3).tolist()} "
        f"motion={motion_options.motion}"
    )
    move_pose_with_singularity_fallback(
        robot,
        waypoint.x_mm,
        waypoint.y_mm,
        waypoint.z_mm,
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
        motion_options,
        allow_clear_confdata_retry=not motion_options.use_current_conf_data,
        allow_movej_singularity_retry=not motion_options.use_current_conf_data,
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
) -> None:
    """Prefer Cartesian loaded travel, but use MoveJ for a proven singular path."""
    rpy_deg = np.asarray(
        [waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg],
        dtype=np.float64,
    )
    print(
        f"Moving loaded transfer {waypoint.name}: "
        f"XYZ(mm)={[round(waypoint.x_mm, 3), round(waypoint.y_mm, 3), round(waypoint.z_mm, 3)]} "
        f"RPY(deg)={rpy_deg.round(3).tolist()} motion=movel"
    )
    try:
        move_pose_with_singularity_fallback(
            robot,
            waypoint.x_mm,
            waypoint.y_mm,
            waypoint.z_mm,
            float(rpy_deg[0]),
            float(rpy_deg[1]),
            float(rpy_deg[2]),
            replace(motion_options, motion="movel", zone_mm=0.0, use_current_conf_data=True),
            allow_clear_confdata_retry=True,
            allow_movej_singularity_retry=True,
        )
    except Exception as exc:
        # A current-conf MoveL can first report -50021; its internal no-conf
        # retry may then expose the real path singularity. Handle that chained
        # case here as well.
        message = str(exc)
        if "50102" not in message and "奇异点" not in message:
            raise
        print(
            f"Loaded MoveL to {waypoint.name} still crosses a singularity after confData retry; "
            "stopping it and using MoveJ without confData."
        )
        robot.stop_motion()
        robot.move_to_pose_mm_deg(
            waypoint.x_mm,
            waypoint.y_mm,
            waypoint.z_mm,
            float(rpy_deg[0]),
            float(rpy_deg[1]),
            float(rpy_deg[2]),
            options=replace(
                motion_options,
                motion="movej",
                zone_mm=0.0,
                use_current_conf_data=False,
            ),
        )


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
) -> None:
    # A* is a physically taught and verified empty/loaded transition point.
    # Use joint interpolation B -> A* -> dynamic A; reserve Cartesian MoveL
    # for the short vertical A -> pickup contact motion.
    approach_pose = (
        float(approach_xyz_mm[0]),
        float(approach_xyz_mm[1]),
        float(approach_xyz_mm[2]),
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
    )
    current_pose = robot.read_current_pose()
    a_star_xyz_mm = np.asarray(
        [WAYPOINT_A_STAR.x_mm, WAYPOINT_A_STAR.y_mm, WAYPOINT_A_STAR.z_mm],
        dtype=np.float64,
    )
    a_star_rotation = rpy_xyz_to_matrix(
        np.radians([WAYPOINT_A_STAR.rx_deg, WAYPOINT_A_STAR.ry_deg, WAYPOINT_A_STAR.rz_deg])
    )
    at_a_star = (
        float(np.linalg.norm(current_pose.translation_mm() - a_star_xyz_mm)) <= 5.0
        and rotation_distance_deg(
            rpy_xyz_to_matrix(current_pose.rpy_rad_xyz),
            a_star_rotation,
        )
        <= 5.0
    )
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
    print(
        "Moving from A* to dynamic A: "
        f"XYZ(mm)={np.asarray(approach_pose[:3]).round(2).tolist()} "
        f"RPY(deg)={np.asarray(approach_pose[3:6]).round(2).tolist()} motion=movej"
    )
    move_pose_with_singularity_fallback(
        robot,
        *approach_pose,
        replace(motion_options, motion="movej", zone_mm=0.0, use_current_conf_data=True),
        allow_clear_confdata_retry=True,
        allow_movej_singularity_retry=False,
    )


def wait_with_operator_stop(duration_s: float, robot: XCoreRobotClient, motion_options: MotionOptions) -> None:
    deadline = time.time() + max(0.0, duration_s)
    while time.time() < deadline:
        if motion_options.stop_requested is not None and motion_options.stop_requested():
            robot.stop_motion()
            raise RuntimeError("Batch interrupted by operator software stop request.")
        time.sleep(min(0.05, max(0.0, deadline - time.time())))


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
    return cups


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
    for cup in suction_cup_specs(args):
        set_suction_output(robot, args, state, do_port=cup.do_port)


def build_suction_approach_plans(
    candidate: ClusterCandidate,
    scene_candidates: list[ClusterCandidate],
    current_tcp_xyz_mm: np.ndarray,
    physical_approach_xyz_mm: np.ndarray,
    physical_pickup_xyz_mm: np.ndarray,
    rpy_candidates: list[tuple[str, np.ndarray]],
    args: argparse.Namespace,
) -> list[SuctionApproachPlan]:
    """Rank TCP targets for both cups while keeping the chosen cup on the package."""
    plans: list[SuctionApproachPlan] = []
    cups = suction_cup_specs(args)
    other_package_points = [
        np.asarray(item.point_base_mm, dtype=np.float64)
        for item in scene_candidates
        # A motion-blocked detection is still a real physical obstacle for
        # the unused suction cup.
        if item.index != candidate.index
    ]
    preferred_clearance_mm = max(0.0, float(args.dual_suction_clearance_mm))
    a_star_xyz_mm = np.asarray(
        [WAYPOINT_A_STAR.x_mm, WAYPOINT_A_STAR.y_mm, WAYPOINT_A_STAR.z_mm],
        dtype=np.float64,
    )

    for rpy_mode, rpy_deg in rpy_candidates:
        rotation = rpy_xyz_to_matrix(np.radians(np.asarray(rpy_deg, dtype=np.float64)))
        for cup in cups:
            selected_offset_base_mm = rotation @ cup.offset_tool_mm
            tcp_approach_xyz_mm = physical_approach_xyz_mm - selected_offset_base_mm
            tcp_pickup_xyz_mm = physical_pickup_xyz_mm - selected_offset_base_mm
            travel_mm = float(np.linalg.norm(tcp_approach_xyz_mm - current_tcp_xyz_mm))
            transition_mm = float(np.linalg.norm(tcp_approach_xyz_mm - a_star_xyz_mm))
            radial_reach_mm = float(np.linalg.norm(tcp_approach_xyz_mm))

            unused_clearance_mm = float("inf")
            for other_cup in cups:
                if other_cup.name == cup.name:
                    continue
                unused_pickup_xyz_mm = tcp_pickup_xyz_mm + rotation @ other_cup.offset_tool_mm
                for other_point_mm in other_package_points:
                    # Ignore only an obstacle whose top is far below the
                    # unused cup. A taller blocked package must never be
                    # ignored merely because the absolute height difference
                    # is large.
                    if float(unused_pickup_xyz_mm[2] - other_point_mm[2]) > 180.0:
                        continue
                    clearance_mm = float(
                        np.linalg.norm(unused_pickup_xyz_mm[:2] - other_point_mm[:2])
                    )
                    unused_clearance_mm = min(unused_clearance_mm, clearance_mm)

            clearance_penalty = 0.0
            if np.isfinite(unused_clearance_mm):
                clearance_penalty = max(0.0, preferred_clearance_mm - unused_clearance_mm) * 8.0

            score = travel_mm + 0.35 * transition_mm + 0.20 * radial_reach_mm + clearance_penalty
            plans.append(
                SuctionApproachPlan(
                    cup=cup,
                    approach_xyz_mm=tcp_approach_xyz_mm,
                    pickup_xyz_mm=tcp_pickup_xyz_mm,
                    rpy_deg=np.asarray(rpy_deg, dtype=np.float64),
                    rpy_mode=rpy_mode,
                    score=score,
                    travel_mm=travel_mm,
                    radial_reach_mm=radial_reach_mm,
                    unused_cup_clearance_mm=unused_clearance_mm,
                )
            )

    primary_mode, primary_rpy_deg = rpy_candidates[0]
    plans.sort(
        key=lambda item: (
            0
            if item.rpy_mode == primary_mode
            and np.allclose(item.rpy_deg, primary_rpy_deg, atol=1e-6)
            else 1,
            item.score,
        )
    )
    fixed_plans = [
        item
        for item in plans
        if item.rpy_mode == primary_mode
        and np.allclose(item.rpy_deg, primary_rpy_deg, atol=1e-6)
    ]
    fallback_plans = [
        item
        for item in plans
        if not (
            item.rpy_mode == primary_mode
            and np.allclose(item.rpy_deg, primary_rpy_deg, atol=1e-6)
        )
    ]
    # Controller IK cannot be queried locally on this robot model, so each
    # extra plan means a real controller rejection/round trip. Keep only the
    # three best reach/path alternatives after both fixed-heading cups fail.
    return fixed_plans + fallback_plans[:3]


def waypoint_for_selected_cup(waypoint: RobotWaypoint, cup: SuctionCupSpec) -> RobotWaypoint:
    """Shift TCP so the selected cup reaches a waypoint taught for the primary cup."""
    rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    tcp_shift_mm = rotation @ cup.offset_tool_mm
    return RobotWaypoint(
        name=f"{waypoint.name}[{cup.name}]",
        x_mm=float(waypoint.x_mm - tcp_shift_mm[0]),
        y_mm=float(waypoint.y_mm - tcp_shift_mm[1]),
        z_mm=float(waypoint.z_mm - tcp_shift_mm[2]),
        rx_deg=waypoint.rx_deg,
        ry_deg=waypoint.ry_deg,
        rz_deg=waypoint.rz_deg,
    )


def cup_center_at_tcp_waypoint(waypoint: RobotWaypoint, cup: SuctionCupSpec) -> np.ndarray:
    rotation = rpy_xyz_to_matrix(
        np.radians([waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg])
    )
    tcp_xyz_mm = np.asarray([waypoint.x_mm, waypoint.y_mm, waypoint.z_mm], dtype=np.float64)
    return tcp_xyz_mm + rotation @ cup.offset_tool_mm


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
    *,
    prefer_original_orientation: bool,
) -> list[RobotWaypoint]:
    """Create same-center functional poses with yaw-only IK alternatives."""
    original_tcp_waypoint = waypoint_for_selected_cup(physical_waypoint, cup)
    if float(np.linalg.norm(cup.offset_tool_mm)) < 1e-6:
        return [original_tcp_waypoint]

    alternatives: list[tuple[float, RobotWaypoint]] = []
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
        score = radial_reach_mm + 0.25 * travel_mm
        alternatives.append((score, tcp_waypoint))

    alternatives.sort(key=lambda item: item[0])
    best_alternatives = [item[1] for item in alternatives[:3]]
    if prefer_original_orientation:
        return [original_tcp_waypoint, *best_alternatives]

    original_xyz_mm = np.asarray(
        [original_tcp_waypoint.x_mm, original_tcp_waypoint.y_mm, original_tcp_waypoint.z_mm],
        dtype=np.float64,
    )
    original_score = float(np.linalg.norm(original_xyz_mm)) + 0.25 * float(
        np.linalg.norm(original_xyz_mm - current_tcp_xyz_mm)
    )
    ranked = [(original_score, original_tcp_waypoint), *alternatives[:3]]
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked]


def move_selected_cup_to_functional_waypoint(
    physical_waypoint: RobotWaypoint,
    cup: SuctionCupSpec,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    *,
    prefer_original_orientation: bool,
) -> tuple[RobotWaypoint, np.ndarray]:
    current_tcp_xyz_mm = robot.read_current_pose().translation_mm()
    candidates = functional_waypoint_candidates_for_cup(
        physical_waypoint,
        cup,
        current_tcp_xyz_mm,
        prefer_original_orientation=prefer_original_orientation,
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
            )
            return tcp_waypoint, cup_center_mm
        except Exception as exc:
            last_error = exc
            if not is_no_ik_solution_error(exc):
                raise
            print(
                f"Functional {physical_waypoint.name} plan #{rank} has no IK; "
                "trying the next yaw-only plan while keeping the same physical center. "
                f"Controller error: {exc}"
            )
    raise RuntimeError(
        f"No reachable {physical_waypoint.name} pose keeps the {cup.name} cup center at the "
        f"required functional point. Last controller error: {last_error}"
    )


def is_no_ik_solution_error(exc: Exception) -> bool:
    message = str(exc)
    return "-50021" in message or "50021" in message


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


def execute_candidate_sequence(
    candidate: ClusterCandidate,
    robot: XCoreRobotClient,
    motion_options: MotionOptions,
    args: argparse.Namespace,
    next_candidate: ClusterCandidate | None = None,
    prepositioned_approach: ApproachPlan | None = None,
    waybill_inspector: AsyncWaybillInspector | None = None,
    scene_candidates: list[ClusterCandidate] | None = None,
) -> tuple[bool, ApproachPlan | None, bool]:
    if not candidate.motion_safe:
        print(f"Candidate #{candidate.index} is blocked for batch motion: {candidate.filter_note}")
        return False, None, False

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
            f"Pickup orientation: prefer fixed tool X heading RZ={FIXED_GRASP_X_YAW_DEG:.2f} deg; "
            "tool Y and package short-edge directions are ignored. If both fixed-heading cup "
            "plans have no IK, limited yaw-only fallbacks are allowed while the suction face "
            "remains aligned down."
        )

        suction_active = False
        selected_suction_name = "primary"
        selected_suction_port = int(args.suction_do_port)
        selected_suction_cup = suction_cup_specs(args)[0]
        selection_origin_xyz_mm = robot.read_current_pose().translation_mm()
        selected_rpy_deg: np.ndarray | None = None
        selected_rpy_mode = rpy_mode
        last_approach_error: Exception | None = None
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
            print(
                "Routing empty tool through B and verified A*, then moving to dynamic A."
            )
            move_waypoint(WAYPOINT_B, robot, approach_options)
            # Candidate orientation sign/rotation limits must be evaluated
            # from the actual B pose.  The pose before routing to B can differ
            # by more than 90 degrees and previously caused a stale no-flip
            # fallback that ignored the YOLO short edge.
            (
                approach_xyz_mm,
                pickup_xyz_mm,
                target_rpy_deg,
                rpy_mode,
                rpy_candidates,
            ) = compute_approach_geometry(candidate, robot, args)
            print(
                "Replanned pickup orientation from the actual B pose: "
                f"RPY(deg)={target_rpy_deg.round(2).tolist()} mode={rpy_mode}"
            )
            suction_plans = build_suction_approach_plans(
                candidate,
                scene_candidates or [candidate],
                selection_origin_xyz_mm,
                approach_xyz_mm,
                pickup_xyz_mm,
                rpy_candidates,
                args,
            )
            for rank, suction_plan in enumerate(suction_plans, start=1):
                try:
                    clearance_text = (
                        "clear"
                        if not np.isfinite(suction_plan.unused_cup_clearance_mm)
                        else f"{suction_plan.unused_cup_clearance_mm:.1f}mm"
                    )
                    print(
                        f"Trying dual-suction plan #{rank}: cup={suction_plan.cup.name} "
                        f"DO{args.suction_do_board}_{suction_plan.cup.do_port} "
                        f"TCP_A(mm)={suction_plan.approach_xyz_mm.round(1).tolist()} "
                        f"RPY(deg)={suction_plan.rpy_deg.round(2).tolist()} "
                        f"travel={suction_plan.travel_mm:.1f}mm "
                        f"reach={suction_plan.radial_reach_mm:.1f}mm "
                        f"unused_clearance={clearance_text} score={suction_plan.score:.1f}"
                    )
                    move_empty_approach_to_a(
                        robot,
                        suction_plan.approach_xyz_mm,
                        suction_plan.rpy_deg,
                        approach_options,
                    )
                    approach_xyz_mm = suction_plan.approach_xyz_mm
                    pickup_xyz_mm = suction_plan.pickup_xyz_mm
                    selected_rpy_deg = suction_plan.rpy_deg
                    selected_rpy_mode = suction_plan.rpy_mode
                    selected_suction_name = suction_plan.cup.name
                    selected_suction_port = suction_plan.cup.do_port
                    selected_suction_cup = suction_plan.cup
                    break
                except Exception as exc:
                    last_approach_error = exc
                    if suction_active:
                        raise
                    print(
                        f"Dual-suction plan #{rank} ({suction_plan.cup.name}) failed before suction; "
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

        # The vertical pickup and retraction are linear whenever the
        # controller accepts the trajectory; the helper falls back to MoveJ
        # only for a controller-reported singularity.
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
        # A* and B are robot/fixture clearance poses taught for the tool4 TCP,
        # not package inspection or placement coordinates. Keep their TCP
        # poses unchanged for either cup. Only functional C and D below are
        # converted so the selected cup/package center reaches the taught
        # physical point.
        for waypoint in (WAYPOINT_A_STAR, WAYPOINT_B):
            raw_rpy_deg = np.asarray(
                [waypoint.rx_deg, waypoint.ry_deg, waypoint.rz_deg],
                dtype=np.float64,
            )
            continuous_rpy_deg = unwrap_rpy_deg(raw_rpy_deg, previous_rpy_deg)
            require_safe_rpy_step(
                previous_rpy_deg,
                continuous_rpy_deg,
                label=f"loaded return waypoint {waypoint.name}",
            )
            print(
                f"Moving loaded return waypoint {waypoint.name}: "
                f"XYZ(mm)={[waypoint.x_mm, waypoint.y_mm, waypoint.z_mm]} "
                f"raw RPY(deg)={raw_rpy_deg.round(2).tolist()} "
                f"continuous RPY(deg)={continuous_rpy_deg.round(2).tolist()} motion=movej"
            )
            reached_pose = robot.move_to_pose_mm_deg(
                waypoint.x_mm,
                waypoint.y_mm,
                waypoint.z_mm,
                float(continuous_rpy_deg[0]),
                float(continuous_rpy_deg[1]),
                float(continuous_rpy_deg[2]),
                options=replace(approach_options, zone_mm=0.0, use_current_conf_data=True),
            )
            previous_rpy_deg = reached_pose.rpy_deg_xyz()
        if waybill_inspector is not None:
            print(
                f"At B; waiting {args.waybill_start_delay_s:.2f}s before starting "
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
            "the selected cup/package center, rather than tool4 TCP, must reach C and D."
        )
        loaded_waypoint_c, selected_cup_center_c_mm = move_selected_cup_to_functional_waypoint(
            WAYPOINT_C,
            selected_suction_cup,
            robot,
            linear_options,
            # Bottom inspection is invariant to in-plane package yaw. Rank
            # reachable TCP geometry first for the offset secondary cup.
            prefer_original_orientation=False,
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

        print("Moving from C to placement point D; D remains a stop/dwell point.")
        loaded_waypoint_d, selected_cup_center_d_mm = move_selected_cup_to_functional_waypoint(
            WAYPOINT_D,
            selected_suction_cup,
            robot,
            stop_options,
            # Preserve the taught placement yaw whenever it is reachable;
            # yaw-only alternatives are a last resort.
            prefer_original_orientation=True,
        )
        print(
            f"At functional placement D for candidate #{candidate.index}; "
            f"{selected_suction_name} cup/package center="
            f"{selected_cup_center_d_mm.round(1).tolist()}; "
            f"disabling {selected_suction_name} suction "
            f"DO{args.suction_do_board}_{selected_suction_port}. "
            f"Waiting {args.place_dwell_s:.1f}s."
        )
        set_all_suction_outputs(robot, args, False)
        suction_active = False
        wait_with_operator_stop(args.place_dwell_s, robot, stop_options)

        next_approach_plan: ApproachPlan | None = None
        if next_candidate is not None and next_candidate.motion_safe:
            (
                next_approach_xyz_mm,
                next_pickup_xyz_mm,
                next_target_rpy_deg,
                next_rpy_mode,
                next_rpy_candidates,
            ) = compute_approach_geometry(next_candidate, robot, args)
            print(
                f"Returning empty through C/B and continuing directly to "
                f"candidate #{next_candidate.index} A point."
            )
            move_waypoint(WAYPOINT_C, robot, approach_options)
            move_waypoint(WAYPOINT_B, robot, approach_options)
            last_next_error: Exception | None = None
            for next_candidate_rpy_mode, next_candidate_rpy_deg in next_rpy_candidates:
                try:
                    print(
                        f"Trying next approach orientation {next_candidate_rpy_mode}: "
                        f"RPY(deg)={next_candidate_rpy_deg.round(2).tolist()}"
                    )
                    move_empty_approach_to_a(
                        robot,
                        next_approach_xyz_mm,
                        next_candidate_rpy_deg,
                        approach_options,
                    )
                    next_approach_plan = ApproachPlan(
                        candidate_index=next_candidate.index,
                        approach_xyz_mm=next_approach_xyz_mm,
                        pickup_xyz_mm=next_pickup_xyz_mm,
                        rpy_deg=next_candidate_rpy_deg,
                        rpy_mode=next_candidate_rpy_mode,
                        suction_name="primary",
                        suction_do_port=int(args.suction_do_port),
                    )
                    break
                except Exception as exc:
                    last_next_error = exc
                    print(
                        f"Next approach orientation {next_candidate_rpy_mode} failed; "
                        "the next package will be approached from the current pose on its own turn."
                    )
            if next_approach_plan is None:
                print(
                    f"Next candidate #{next_candidate.index} was not prepositioned; "
                    "the batch will continue with a fresh approach on that candidate."
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
) -> bool:
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
        return True

    motion_candidates = [candidate for candidate in candidates if candidate.motion_safe]
    skipped_count = len(candidates) - len(motion_candidates)
    if skipped_count:
        print(f"Skipping {skipped_count} blocked candidate(s); only motion-safe packages will be executed.")
    if not motion_candidates:
        print("No motion-safe candidate is available for robot motion.")
        return False

    # The scene becomes stale as soon as one package is removed. Try ranked
    # candidates only until one succeeds, then force a fresh YOLO/depth pass.
    for candidate in motion_candidates:
        print(
            f"Next-pick choice #{candidate.index}: Z={candidate.point_base_mm[2]:.1f} mm "
            f"confidence={candidate.confidence:.2f} flatness={candidate.flatness_mm:.2f} mm."
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
        )
        if not ok:
            return False
        if completed:
            print("One package completed; forcing fresh YOLO/depth analysis before choosing another.")
            return True
    else:
        print("No package completed successfully; automatic retry is disabled to avoid an endless loop.")
        return False


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
            f"surface XYZ(mm)={candidate.point_base_mm.round(1).tolist()}"
        )
        if candidate.short_axis_base is not None:
            print(
                f"      YOLO short edge (base)={candidate.short_axis_base.round(4).tolist()} "
                "(reported for diagnostics only; not used for grasp orientation)"
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
    if not np.isfinite(args.secondary_suction_offset_y_mm):
        parser.error("--secondary-suction-offset-y-mm must be finite")
    if args.dual_suction_clearance_mm < 0.0 or not np.isfinite(args.dual_suction_clearance_mm):
        parser.error("--dual-suction-clearance-mm must be a non-negative finite value")
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
    if args.waybill_capture_duration_s <= args.waybill_c_settle_s + args.waybill_post_c_capture_s:
        parser.error("--waybill-capture-duration-s must exceed C settle plus static capture time")
    if not args.disable_secondary_suction and args.secondary_suction_do_port == args.suction_do_port:
        parser.error("primary and secondary suction cups must use different DO ports")
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
            output_dir=args.waybill_output_dir,
            confidence=args.waybill_conf,
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
    operator_stop_requested = False

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

    cv2.namedWindow(COLOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(COLOR_WINDOW, mouse_callback)
    cv2.setMouseCallback(DEPTH_WINDOW, mouse_callback)

    def request_operator_stop(reason: str = "SPACE") -> None:
        nonlocal operator_stop_requested, batch_auto_enabled
        if not operator_stop_requested:
            print(f"{reason} pressed: requesting robot software stop and aborting the current batch.")
        operator_stop_requested = True
        batch_auto_enabled = False

    def request_quit() -> None:
        nonlocal quit_requested
        quit_requested = True
        request_operator_stop("Q")

    def poll_operator_stop_key() -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            request_operator_stop()
        elif key == ord("q"):
            request_quit()
        return operator_stop_requested

    motion_options.stop_requested = poll_operator_stop_key

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
        batch_ok = execute_all_candidates(
            analysis.candidates,
            robot,
            motion_options,
            args.dry_run or robot is None,
            args,
            waybill_inspector=waybill_inspector,
        )
        if not batch_ok:
            batch_auto_enabled = False
            print(
                "Continuous batch mode disabled after a motion error. "
                "Resolve the robot/suction state and press Enter to start again."
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
        return batch_ok

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
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    request_quit()
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

            cv2.imshow(COLOR_WINDOW, color_canvas)
            cv2.imshow(DEPTH_WINDOW, depth_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                request_quit()
                break
            if key == 32:
                request_operator_stop()
                if robot is not None:
                    robot.stop_motion()
                continue
            if key in (ord("m"), ord("b"), ord("1"), ord("2"), ord("3"), ord("4")):
                if key == ord("m"):
                    roi_edit_target = "overall"
                elif key == ord("b"):
                    roi_edit_target = "exclude"
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
                if roi_config.overall_polygon is None and not roi_config.support_polygons:
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
        cv2.destroyAllWindows()
        camera.stop()
        if waybill_inspector is not None:
            waybill_inspector.close()
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
