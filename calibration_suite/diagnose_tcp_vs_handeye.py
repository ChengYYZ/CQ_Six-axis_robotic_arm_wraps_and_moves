from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from project0714_calib.orbbec_camera import CameraOpenOptions, OrbbecRGBDCamera
from project0714_calib.xcore_robot import XCoreRobotClient
from surface_cluster_grasp import (
    camera_point_to_base_mm,
    load_camera_matrix,
    load_camera_point_to_base_transform,
)


WINDOW = "TCP vs hand-eye diagnosis (read-only)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate fixed hand-eye XY bias from tool-rotating TCP XY bias without commanding robot motion."
    )
    parser.add_argument(
        "--intrinsics",
        default="calibration_suite/workspace/intrinsics/camera_intrinsics.json",
    )
    parser.add_argument(
        "--hand-eye",
        default="calibration_suite/workspace/eye_to_hand/eye_to_hand_result.json",
    )
    parser.add_argument("--robot-ip", default="192.168.2.160")
    parser.add_argument("--depth-radius-px", type=int, default=4)
    parser.add_argument("--output", default="calibration_suite/workspace/tcp_handeye_diagnosis.json")
    return parser


def pixel_point_camera_mm(
    pixel: tuple[int, int], depth_mm: np.ndarray, camera_matrix: np.ndarray, radius: int
) -> np.ndarray:
    x, y = pixel
    h, w = depth_mm.shape[:2]
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    values = depth_mm[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        raise RuntimeError("No valid depth around the selected marker.")
    z = float(np.median(values))
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    return np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64)


def solve_xy(marker_base_mm: np.ndarray, samples: list[dict]) -> dict:
    # reported_TCP_xy - camera_marker_xy = fixed_handeye_xy + R_tool_xy * tcp_offset_xy
    rows: list[np.ndarray] = []
    values: list[float] = []
    for sample in samples:
        rotation = np.asarray(sample["rotation"], dtype=np.float64)
        error_xy = np.asarray(sample["tcp_base_mm"], dtype=np.float64)[:2] - marker_base_mm[:2]
        block = rotation[:2, :2]
        rows.extend(
            [
                np.asarray([1.0, 0.0, block[0, 0], block[0, 1]]),
                np.asarray([0.0, 1.0, block[1, 0], block[1, 1]]),
            ]
        )
        values.extend(error_xy.tolist())
    design = np.vstack(rows)
    observation = np.asarray(values, dtype=np.float64)
    solution, _residuals, rank, singular_values = np.linalg.lstsq(design, observation, rcond=None)
    predicted = design @ solution
    rms = float(np.sqrt(np.mean((predicted - observation) ** 2)))
    return {
        "handeye_fixed_xy_mm": solution[:2].round(3).tolist(),
        "tcp_tool_xy_mm": solution[2:].round(3).tolist(),
        "fit_rms_mm": round(rms, 3),
        "rank": int(rank),
        "singular_values": singular_values.round(6).tolist(),
        "well_conditioned": bool(rank == 4),
    }


def main() -> int:
    args = build_parser().parse_args()
    camera_matrix = load_camera_matrix(args.intrinsics)
    camera_to_base = load_camera_point_to_base_transform(args.hand_eye)
    camera = OrbbecRGBDCamera(CameraOpenOptions())
    robot = XCoreRobotClient(args.robot_ip)
    selected_pixel: tuple[int, int] | None = None
    marker_base_mm: np.ndarray | None = None
    samples: list[dict] = []

    def on_mouse(event, x, y, _flags, _param):
        nonlocal selected_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixel = (int(x), int(y))
            print(f"Marker pixel selected: {selected_pixel}. Press M to confirm it.")

    camera.start()
    robot.connect()  # Read-only: no operating-mode, power, or motion command is sent.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("This tool never commands robot motion.")
    print("1) Move the robot away; click one fixed flat marker, then press M.")
    print("2) Manually align the real suction center to that marker and press S.")
    print("3) Repeat at >=2 substantially different tool yaw angles (ideally ~180 deg apart).")
    print("Keys: M=set marker, S=sample current robot pose, C=calculate, Q=quit.")
    latest_depth: np.ndarray | None = None
    try:
        while True:
            frames = camera.get_frames(500)
            if frames is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            color, latest_depth, _depth_display = frames
            canvas = color.copy()
            if selected_pixel is not None:
                cv2.drawMarker(canvas, selected_pixel, (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
            text = f"marker={'set' if marker_base_mm is not None else 'unset'} samples={len(samples)}"
            cv2.putText(canvas, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("m"), ord("M")):
                if selected_pixel is None or latest_depth is None:
                    print("Click the fixed marker first.")
                    continue
                point_camera = pixel_point_camera_mm(
                    selected_pixel, latest_depth, camera_matrix, max(1, args.depth_radius_px)
                )
                marker_base_mm = camera_point_to_base_mm(camera_to_base, point_camera)
                samples.clear()
                print(f"Marker base XYZ(mm)={marker_base_mm.round(2).tolist()}")
            if key in (ord("s"), ord("S")):
                if marker_base_mm is None:
                    print("Set the marker with M first.")
                    continue
                pose = robot.read_current_pose()
                sample = {
                    "tcp_base_mm": pose.translation_mm().tolist(),
                    "rpy_deg": pose.rpy_deg_xyz().tolist(),
                    "rotation": pose.to_transform()[:3, :3].tolist(),
                }
                samples.append(sample)
                print(
                    f"Sample #{len(samples)} TCP XYZ(mm)={pose.translation_mm().round(2).tolist()} "
                    f"RPY(deg)={pose.rpy_deg_xyz().round(2).tolist()}"
                )
            if key in (ord("c"), ord("C")):
                if marker_base_mm is None or len(samples) < 2:
                    print("At least two aligned samples at different yaw angles are required.")
                    continue
                result = solve_xy(marker_base_mm, samples)
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "marker_base_mm": marker_base_mm.tolist(),
                    "samples": samples,
                    "result": result,
                    "interpretation": {
                        "handeye_fixed_xy_mm": "Error fixed in the robot base/camera direction.",
                        "tcp_tool_xy_mm": "Error rotating with the tool; correct the TCP XY definition.",
                    },
                }
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"Saved: {output.resolve()}")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
