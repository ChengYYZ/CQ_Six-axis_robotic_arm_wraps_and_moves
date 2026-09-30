from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project0714_calib.orbbec_camera import CameraOpenOptions, OrbbecRGBDCamera  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an empty platform with Orbbec RGB-D, select the platform ROI, "
            "and export median depth plus camera/base-frame PLY point clouds."
        )
    )
    parser.add_argument("--frames", type=int, default=40, help="Number of RGB-D frames to capture")
    parser.add_argument("--interval-s", type=float, default=0.10, help="Delay between retained frames")
    parser.add_argument("--align-mode", choices=("sw", "hw", "off"), default="sw")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--intrinsics",
        default=str(ROOT / "workspace" / "intrinsics" / "camera_intrinsics.json"),
        help="Camera matrix JSON used by the grasp pipeline",
    )
    parser.add_argument(
        "--hand-eye",
        default=str(ROOT / "workspace" / "eye_to_hand" / "eye_to_hand_result.json"),
        help="Camera-to-robot-base transform JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "workspace" / "empty_platform_baselines"),
        help="Parent directory for this timestamped capture",
    )
    parser.add_argument("--min-depth-mm", type=float, default=300.0)
    parser.add_argument("--max-depth-mm", type=float, default=2500.0)
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.60,
        help="Minimum fraction of captured frames with valid depth for each baseline pixel",
    )
    parser.add_argument(
        "--save-raw-frames",
        action="store_true",
        help="Also save every aligned depth frame as compressed NPZ (larger output)",
    )
    return parser


def load_intrinsics(path: str | Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
    if matrix.shape != (3, 3) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"Invalid camera_matrix in {path}")
    return matrix


def load_camera_to_base(path: str | Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    block = payload.get("base_to_camera", payload.get("camera_to_base"))
    if not isinstance(block, dict):
        raise KeyError(f"No base_to_camera/camera_to_base transform in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(block["rotation_matrix"], dtype=np.float64)
    transform[:3, 3] = np.asarray(block["translation_m"], dtype=np.float64).reshape(3)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"Invalid camera-to-base transform in {path}")
    return transform


def resize_depth_to_color(depth_mm: np.ndarray, color_shape: tuple[int, int]) -> np.ndarray:
    height, width = color_shape
    if depth_mm.shape == (height, width):
        return depth_mm.astype(np.float32, copy=False)
    return cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def wait_for_rgbd_frame(
    camera: OrbbecRGBDCamera,
    timeout_ms: int,
    *,
    max_wait_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Poll through temporarily incomplete color/depth frames until a pair arrives."""
    deadline = time.monotonic() + max(0.1, max_wait_s)
    while time.monotonic() < deadline:
        frames = camera.get_frames(min(max(1, timeout_ms), 500))
        if frames is not None:
            return frames
        time.sleep(0.02)
    return None


class PolygonPicker:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points: list[tuple[int, int]] = []
        self.confirmed = False
        self.cancelled = False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def draw(self, image: np.ndarray) -> np.ndarray:
        view = image.copy()
        if len(self.points) > 1:
            cv2.polylines(view, [np.asarray(self.points, dtype=np.int32)], False, (0, 255, 0), 2)
        for point in self.points:
            cv2.circle(view, point, 5, (0, 0, 255), -1)
        cv2.putText(
            view,
            "L-click vertices | R-click undo | C/Enter confirm | R reset | Esc cancel",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            "L-click vertices | R-click undo | C/Enter confirm | R reset | Esc cancel",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return view


def select_platform_roi(color: np.ndarray) -> np.ndarray:
    window_name = "Empty platform ROI selection"
    picker = PolygonPicker(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, picker.on_mouse)
    try:
        while True:
            cv2.imshow(window_name, picker.draw(color))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("c"), 13, 10):
                if len(picker.points) >= 3:
                    picker.confirmed = True
                    break
                print("Select at least 3 platform boundary points before confirming.")
            elif key == ord("r"):
                picker.points.clear()
            elif key == 27:
                picker.cancelled = True
                break
    finally:
        cv2.destroyWindow(window_name)
    if not picker.confirmed:
        raise RuntimeError("Platform ROI selection was cancelled.")
    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(picker.points, dtype=np.int32)], 255)
    if int(np.count_nonzero(mask)) < 1000:
        raise ValueError("Selected platform ROI is too small; capture cancelled.")
    return mask > 0


def make_cloud(depth_mm: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask & np.isfinite(depth_mm) & (depth_mm > 0.0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth_mm[ys, xs].astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32)


def write_colored_ply(path: Path, points_mm: np.ndarray, colors_bgr: np.ndarray) -> None:
    points = np.asarray(points_mm, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
    if len(points) != len(colors):
        raise ValueError("Point and color counts do not match")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertices = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.interval_s < 0 or args.timeout_ms <= 0:
        parser.error("--interval-s must be non-negative and --timeout-ms positive")
    if args.min_depth_mm <= 0 or args.max_depth_mm <= args.min_depth_mm:
        parser.error("Depth limits must satisfy 0 < MIN < MAX")
    if not 0.0 < args.min_valid_ratio <= 1.0:
        parser.error("--min-valid-ratio must be in (0, 1]")

    intrinsics = load_intrinsics(args.intrinsics)
    camera_to_base = load_camera_to_base(args.hand_eye)
    capture_dir = Path(args.output_dir) / datetime.now().strftime("empty_platform_%Y%m%d_%H%M%S")
    capture_dir.mkdir(parents=True, exist_ok=False)
    camera = OrbbecRGBDCamera(
        CameraOpenOptions(align_mode=args.align_mode, wait_timeout_ms=args.timeout_ms)
    )
    captured_depths: list[np.ndarray] = []
    color = None
    camera.start()
    try:
        print("Camera started. Clear the platform and keep the camera/platform fixed.")
        # Drain startup exposure/auto-alignment frames before presenting the ROI image.
        for _ in range(8):
            wait_for_rgbd_frame(camera, args.timeout_ms, max_wait_s=2.0)
        preview = wait_for_rgbd_frame(camera, args.timeout_ms, max_wait_s=15.0)
        if preview is None:
            raise RuntimeError("No RGB-D frame received from Orbbec camera.")
        color, first_depth, _ = preview
        first_depth = resize_depth_to_color(first_depth, color.shape[:2])
        roi_mask = select_platform_roi(color)

        window_name = "Empty platform capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            for index in range(args.frames):
                frames = wait_for_rgbd_frame(camera, args.timeout_ms, max_wait_s=5.0)
                if frames is None:
                    print(f"Frame {index + 1}/{args.frames}: no synchronized frame, skipping.")
                    continue
                color, depth_mm, _depth_display = frames
                depth_mm = resize_depth_to_color(depth_mm, color.shape[:2])
                valid = (
                    np.isfinite(depth_mm)
                    & (depth_mm >= args.min_depth_mm)
                    & (depth_mm <= args.max_depth_mm)
                )
                captured_depths.append(np.where(valid, depth_mm, np.nan).astype(np.float32))
                preview_image = color.copy()
                overlay = preview_image.copy()
                cv2.polylines(
                    overlay,
                    [cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]],
                    True,
                    (0, 255, 0),
                    2,
                )
                cv2.addWeighted(overlay, 0.65, preview_image, 0.35, 0.0, preview_image)
                cv2.putText(
                    preview_image,
                    f"Captured {len(captured_depths)}/{args.frames}  |  q aborts",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise RuntimeError("Capture aborted by operator.")
                print(f"Captured frame {len(captured_depths)}/{args.frames}")
                if args.interval_s:
                    time.sleep(args.interval_s)
        finally:
            cv2.destroyWindow(window_name)
    except Exception:
        # Keep failed/aborted captures recognizable, but don't mistake them for a baseline.
        raise
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if len(captured_depths) < 3:
        raise RuntimeError(f"Only {len(captured_depths)} valid frames were captured; need at least 3.")
    depth_stack = np.stack(captured_depths, axis=0)
    valid_count = np.sum(np.isfinite(depth_stack), axis=0)
    required_count = int(np.ceil(len(captured_depths) * args.min_valid_ratio))
    stable_mask = roi_mask & (valid_count >= required_count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_depth = np.nanmedian(depth_stack, axis=0).astype(np.float32)
    median_depth[~stable_mask] = 0.0
    if args.save_raw_frames:
        np.savez_compressed(capture_dir / "aligned_depth_frames_mm.npz", depth_mm=depth_stack)
    del depth_stack

    points_camera_mm = make_cloud(median_depth, stable_mask, intrinsics)
    colors_bgr = color[stable_mask & (median_depth > 0.0)]
    if len(points_camera_mm) < 1000:
        raise RuntimeError(
            f"Only {len(points_camera_mm)} stable platform points remain; check ROI/depth alignment."
        )
    points_base_mm = (camera_to_base @ np.column_stack(
        (points_camera_mm.astype(np.float64) / 1000.0, np.ones(len(points_camera_mm)))
    ).T).T[:, :3] * 1000.0

    cv2.imwrite(str(capture_dir / "empty_platform_rgb.png"), color)
    cv2.imwrite(str(capture_dir / "platform_roi_mask.png"), roi_mask.astype(np.uint8) * 255)
    cv2.imwrite(
        str(capture_dir / "empty_platform_depth_preview.png"),
        cv2.normalize(median_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
    )
    np.save(capture_dir / "empty_platform_depth_mm.npy", median_depth)
    np.save(capture_dir / "stable_valid_pixel_count.npy", valid_count.astype(np.uint16))
    write_colored_ply(capture_dir / "empty_platform_camera_mm.ply", points_camera_mm, colors_bgr)
    write_colored_ply(capture_dir / "empty_platform_base_mm.ply", points_base_mm, colors_bgr)

    manifest = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "coordinate_frames": {
            "empty_platform_camera_mm.ply": "Orbbec camera frame, millimetres",
            "empty_platform_base_mm.ply": "robot base frame, millimetres",
            "empty_platform_depth_mm.npy": "aligned depth in millimetres; 0 means invalid/outside stable ROI",
        },
        "frames_requested": int(args.frames),
        "frames_captured": len(captured_depths),
        "alignment_mode": args.align_mode,
        "image_width": int(color.shape[1]),
        "image_height": int(color.shape[0]),
        "depth_limits_mm": [float(args.min_depth_mm), float(args.max_depth_mm)],
        "minimum_valid_frame_ratio": float(args.min_valid_ratio),
        "required_valid_frame_count": required_count,
        "stable_platform_points": int(len(points_camera_mm)),
        "roi_polygon_pixels": cv2.findContours(
            roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0][0].reshape(-1, 2).astype(int).tolist(),
        "camera_matrix": intrinsics.tolist(),
        "camera_to_base_transform": camera_to_base.tolist(),
        "intrinsics_file": str(Path(args.intrinsics).resolve()),
        "hand_eye_file": str(Path(args.hand_eye).resolve()),
    }
    (capture_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Empty platform baseline saved to: {capture_dir}")
    print(f"Stable ROI points: {len(points_camera_mm):,}")
    print(f"Camera-frame cloud: {capture_dir / 'empty_platform_camera_mm.ply'}")
    print(f"Robot-base cloud: {capture_dir / 'empty_platform_base_mm.ply'}")
    print(f"Median aligned depth: {capture_dir / 'empty_platform_depth_mm.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
