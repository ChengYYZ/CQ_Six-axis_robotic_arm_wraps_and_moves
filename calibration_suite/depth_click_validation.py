from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import cv2
import numpy as np

from project0714_calib.common import load_json, make_transform


MIN_DEPTH_MM = 20.0
MAX_DEPTH_MM = 10000.0
COLOR_WINDOW = "Project0714 Color"
DEPTH_WINDOW = "Project0714 Depth"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "workspace" / "validation" / "depth_click_validation.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual validation tool: click a pixel and convert it to camera/base coordinates"
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
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=800, help="Color stream height")
    parser.add_argument("--fps", type=int, default=10, help="Stream FPS")
    parser.add_argument("--depth-width", type=int, default=640, help="Depth stream width")
    parser.add_argument("--depth-height", type=int, default=400, help="Depth stream height")
    parser.add_argument("--depth-fps", type=int, default=5, help="Depth stream FPS")
    parser.add_argument("--wait-timeout-ms", type=int, default=500, help="Frame wait timeout in milliseconds")
    parser.add_argument("--radius", type=int, default=2, help="Neighborhood radius used to search valid depth")
    parser.add_argument(
        "--align-mode",
        choices=("sw", "hw", "off"),
        default="sw",
        help="Depth-to-color alignment mode",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_PATH),
        help="Diagnostic log file path",
    )
    return parser


def load_camera_matrix(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    return np.asarray(payload["camera_matrix"], dtype=np.float64)


def load_base_to_camera(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    if "base_to_camera" not in payload:
        raise KeyError("base_to_camera is missing from the hand-eye result JSON.")

    rotation = np.asarray(payload["base_to_camera"]["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(payload["base_to_camera"]["translation_m"], dtype=np.float64)
    return make_transform(rotation, translation)


def frame_to_bgr_image(frame) -> np.ndarray | None:
    from pyorbbecsdk import OBFormat

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()

    if fmt == OBFormat.RGB:
        image = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        image = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape((height, width, 3))
        return image.copy()
    if fmt == OBFormat.MJPG:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        image = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        image = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    return None


def find_valid_depth(depth_mm: np.ndarray, x: int, y: int, radius: int) -> tuple[float | None, tuple[int, int] | None]:
    h, w = depth_mm.shape
    x0 = max(0, x - radius)
    x1 = min(w - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(h - 1, y + radius)

    patch = depth_mm[y0 : y1 + 1, x0 : x1 + 1]
    valid_mask = (patch > MIN_DEPTH_MM) & (patch < MAX_DEPTH_MM)
    if not np.any(valid_mask):
        return None, None

    valid_values = patch[valid_mask]
    median_depth = float(np.median(valid_values))

    best_xy = None
    best_diff = float("inf")
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            value = float(depth_mm[yy, xx])
            if value <= MIN_DEPTH_MM or value >= MAX_DEPTH_MM:
                continue
            diff = abs(value - median_depth)
            if diff < best_diff:
                best_diff = diff
                best_xy = (xx, yy)

    return median_depth, best_xy


def pixel_to_camera(camera_matrix: np.ndarray, x: int, y: int, depth_mm: float) -> np.ndarray:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    z_m = depth_mm / 1000.0
    x_m = (x - cx) * z_m / fx
    y_m = (y - cy) * z_m / fy
    return np.asarray([x_m, y_m, z_m], dtype=np.float64)


def camera_to_base_point(base_to_camera: np.ndarray, point_camera_m: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point_camera_m
    point_base = base_to_camera @ homogeneous
    return point_base[:3]


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    clipped = np.where((depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM), depth_mm, 0)
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored


def draw_status(image: np.ndarray, message_lines: list[str], marker: tuple[int, int] | None) -> np.ndarray:
    canvas = image.copy()
    if marker is not None:
        cv2.drawMarker(
            canvas,
            marker,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.circle(canvas, marker, 18, (255, 255, 255), 1)

    y = 30
    for line in message_lines:
        cv2.putText(
            canvas,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return canvas


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def configure_pipeline(
    color_width: int,
    color_height: int,
    color_fps: int,
    depth_width: int,
    depth_height: int,
    depth_fps: int,
    align_mode: str,
    log_path: Path,
):
    from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline

    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    color_profile = None
    for fmt_name in ("MJPG", "RGB888", "BGRA", "YUYV"):
        try:
            color_profile = color_profiles.get_video_stream_profile(
                color_width,
                color_height,
                getattr(OBFormat, fmt_name),
                color_fps,
            )
            if color_profile is not None:
                break
        except Exception:
            continue
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()

    depth_profile = None
    for candidate in (
        (depth_width, depth_height, depth_fps),
        (depth_width, depth_height, 10),
        (1280, 800, 5),
        (640, 400, 10),
        (640, 400, 5),
    ):
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                candidate[0],
                candidate[1],
                OBFormat.Y16,
                candidate[2],
            )
            if depth_profile is not None:
                break
        except Exception:
            continue
    if depth_profile is None:
        depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    if align_mode == "sw":
        config.set_align_mode(OBAlignMode.SW_MODE)
        append_log(log_path, "Using software depth-to-color alignment.")
    elif align_mode == "hw":
        config.set_align_mode(OBAlignMode.HW_MODE)
        append_log(log_path, "Using hardware depth-to-color alignment.")
    else:
        config.set_align_mode(OBAlignMode.DISABLE)
        append_log(log_path, "Depth-to-color alignment disabled.")

    try:
        pipeline.enable_frame_sync()
    except Exception:
        append_log(log_path, "Frame sync is not available on this device/profile combination.")

    pipeline.start(config)
    append_log(log_path, f"Selected color profile: {color_profile}")
    append_log(log_path, f"Selected depth profile: {depth_profile}")
    return pipeline


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    log_path = Path(args.log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    append_log(log_path, "Starting depth click validation tool.")

    camera_matrix = load_camera_matrix(args.intrinsics)
    transform_base_to_camera = load_base_to_camera(args.hand_eye)

    pipeline = configure_pipeline(
        args.width,
        args.height,
        args.fps,
        args.depth_width,
        args.depth_height,
        args.depth_fps,
        args.align_mode,
        log_path,
    )

    latest = {
        "pixel": None,
        "depth_mm": None,
        "camera_m": None,
        "base_m": None,
        "marker": None,
    }

    def handle_click(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if current_depth_mm is None:
            append_log(log_path, f"Clicked ({x}, {y}) before a valid depth frame arrived.")
            return

        depth_mm, marker = find_valid_depth(current_depth_mm, x, y, args.radius)
        if depth_mm is None or marker is None:
            latest["pixel"] = (x, y)
            latest["depth_mm"] = None
            latest["camera_m"] = None
            latest["base_m"] = None
            latest["marker"] = (x, y)
            append_log(log_path, f"Clicked ({x}, {y}) but no valid depth was found nearby.")
            return

        camera_point_m = pixel_to_camera(camera_matrix, marker[0], marker[1], depth_mm)
        base_point_m = camera_to_base_point(transform_base_to_camera, camera_point_m)

        latest["pixel"] = marker
        latest["depth_mm"] = depth_mm
        latest["camera_m"] = camera_point_m
        latest["base_m"] = base_point_m
        latest["marker"] = marker

        append_log(log_path, "")
        append_log(log_path, f"Pixel              : {marker}")
        append_log(log_path, f"Depth              : {depth_mm:.2f} mm")
        append_log(
            log_path,
            "Camera XYZ (mm)    : "
            f"[{camera_point_m[0] * 1000.0:.3f}, {camera_point_m[1] * 1000.0:.3f}, {camera_point_m[2] * 1000.0:.3f}]",
        )
        append_log(
            log_path,
            "Base XYZ (mm)      : "
            f"[{base_point_m[0] * 1000.0:.3f}, {base_point_m[1] * 1000.0:.3f}, {base_point_m[2] * 1000.0:.3f}]",
        )
        append_log(log_path, "Note               : This tool outputs point coordinates only, not full orientation.")

    cv2.setNumThreads(1)
    cv2.startWindowThread()
    cv2.namedWindow(COLOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(COLOR_WINDOW, handle_click)
    cv2.setMouseCallback(DEPTH_WINDOW, handle_click)
    append_log(log_path, "OpenCV windows created. Entering main loop.")

    current_depth_mm: np.ndarray | None = None

    try:
        frame_counter = 0
        while True:
            try:
                frames = pipeline.wait_for_frames(args.wait_timeout_ms)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    append_log(log_path, f"Unsupported color frame format: {color_frame.get_format()}")
                    continue

                depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
                current_depth_mm = depth_raw.astype(np.float32) * depth_frame.get_depth_scale()
                depth_view = colorize_depth(current_depth_mm)

                message_lines = [
                    "Left click a point in color/depth view",
                    "Keys: q quit, c clear marker",
                ]
                if latest["pixel"] is not None:
                    if latest["depth_mm"] is None:
                        message_lines.append(f"pixel={latest['pixel']} depth=INVALID")
                    else:
                        cam_mm = latest["camera_m"] * 1000.0
                        base_mm = latest["base_m"] * 1000.0
                        message_lines.append(f"pixel={latest['pixel']} depth={latest['depth_mm']:.1f} mm")
                        message_lines.append(
                            f"camera(mm)=({cam_mm[0]:.1f}, {cam_mm[1]:.1f}, {cam_mm[2]:.1f})"
                        )
                        message_lines.append(
                            f"base(mm)=({base_mm[0]:.1f}, {base_mm[1]:.1f}, {base_mm[2]:.1f})"
                        )

                color_canvas = draw_status(color_image, message_lines, latest["marker"])
                depth_canvas = draw_status(depth_view, message_lines, latest["marker"])

                cv2.imshow(COLOR_WINDOW, color_canvas)
                cv2.imshow(DEPTH_WINDOW, depth_canvas)

                frame_counter += 1
                if frame_counter == 1:
                    append_log(log_path, "First frame displayed successfully.")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    append_log(log_path, "Quit requested by keyboard.")
                    break
                if key == ord("c"):
                    latest = {
                        "pixel": None,
                        "depth_mm": None,
                        "camera_m": None,
                        "base_m": None,
                        "marker": None,
                    }
                    append_log(log_path, "Marker cleared.")
            except Exception:
                append_log(log_path, "Frame processing error:")
                append_log(log_path, traceback.format_exc().rstrip())
                time.sleep(0.1)
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        append_log(log_path, "Pipeline stopped and windows destroyed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
