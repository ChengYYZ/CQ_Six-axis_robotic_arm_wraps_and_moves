from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project0714_calib.common import (
    SampleSummary,
    chessboard_object_points,
    draw_chessboard_overlay,
    ensure_dir,
    find_chessboard_corners,
    invert_transform,
    load_json,
    rvec_tvec_to_transform,
    save_json,
    solve_pnp,
    split_transform,
    timestamp_tag,
    transform_to_dict,
)
from project0714_calib.orbbec_camera import CameraOpenOptions, OrbbecColorCamera
from project0714_calib.xcore_robot import RobotPose, XCoreRobotClient


HAND_EYE_METHODS = {
    "TSAI": "CALIB_HAND_EYE_TSAI",
    "PARK": "CALIB_HAND_EYE_PARK",
    "HORAUD": "CALIB_HAND_EYE_HORAUD",
    "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
    "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project0714 eye-to-hand calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture eye-to-hand samples")
    add_board_args(capture)
    add_camera_args(capture)
    capture.add_argument("--robot-ip", required=True, help="Robot controller IP address")
    capture.add_argument(
        "--save-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "eye_to_hand" / "samples"),
        help="Directory used to save captured samples",
    )

    solve = subparsers.add_parser("solve", help="Solve eye-to-hand extrinsics from samples")
    add_board_args(solve)
    solve.add_argument(
        "--sample-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "eye_to_hand" / "samples"),
        help="Directory containing captured samples",
    )
    solve.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON path")
    solve.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "workspace" / "eye_to_hand" / "eye_to_hand_result.json"),
        help="Output JSON path",
    )
    solve.add_argument(
        "--method",
        choices=sorted(HAND_EYE_METHODS),
        default="TSAI",
        help="OpenCV hand-eye solving method",
    )
    return parser


def add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cols", type=int, required=True, help="Number of inner corners per row")
    parser.add_argument("--rows", type=int, required=True, help="Number of inner corners per column")
    parser.add_argument("--square-size-mm", type=float, required=True, help="Chessboard square size in mm")


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=800, help="Camera height")
    parser.add_argument("--fps", type=int, default=10, help="Camera FPS")


def run_capture(args: argparse.Namespace) -> int:
    import cv2

    save_dir = ensure_dir(Path(args.save_dir))
    camera = OrbbecColorCamera(CameraOpenOptions(args.width, args.height, args.fps))
    robot = XCoreRobotClient(args.robot_ip)
    captured = 0
    robot_status = "CONNECTING"

    try:
        camera.start()
        try:
            robot.connect()
            robot_status = "CONNECTED"
            print(f"Robot connected: {args.robot_ip}")
        except Exception as exc:
            robot_status = "CONNECT FAILED"
            print(f"Failed to connect robot {args.robot_ip}: {exc}")
            print("Preview will stay open. Fix robot connection and press r to retry.")

        while True:
            frame = camera.get_bgr_frame(timeout_ms=200)
            if frame is None:
                continue

            found, corners, _ = find_chessboard_corners(frame, args.cols, args.rows)
            overlay = draw_chessboard_overlay(frame, args.cols, args.rows, corners, found)
            status_line = f"saved={captured}  detected={'YES' if found else 'NO'}  key:s save  r retry  q quit"
            cv2.putText(
                overlay,
                status_line,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 120, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"robot: {robot_status}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Project0714 Eye-To-Hand Capture", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                try:
                    robot.connect()
                    robot_status = "CONNECTED"
                    print(f"Robot connected: {args.robot_ip}")
                except Exception as exc:
                    robot_status = "CONNECT FAILED"
                    print(f"Failed to connect robot {args.robot_ip}: {exc}")
                continue
            if key == ord("s"):
                if not found:
                    print("Current frame does not contain a complete chessboard. Skipped.")
                    continue

                try:
                    pose = robot.read_current_pose()
                    robot_status = "CONNECTED"
                except Exception as exc:
                    robot_status = "CONNECT FAILED"
                    print(f"Failed to read robot pose: {exc}")
                    print("Preview remains available. Check robot connection, then press r to retry.")
                    continue

                stem = f"sample_{timestamp_tag()}"
                image_path = save_dir / f"{stem}.png"
                pose_path = save_dir / f"{stem}.json"
                cv2.imwrite(str(image_path), frame)
                save_json(pose_path, pose.to_dict())
                captured += 1
                print(f"[{captured}] Saved image: {image_path}")
                print(f"[{captured}] Saved pose : {pose_path}")
    finally:
        robot.disconnect()
        camera.stop()
        cv2.destroyAllWindows()

    return 0


def run_solve(args: argparse.Namespace) -> int:
    import cv2

    sample_dir = Path(args.sample_dir)
    intrinsics = load_json(args.intrinsics)
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    object_points = chessboard_object_points(args.cols, args.rows, args.square_size_mm)

    image_paths = sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No eye-to-hand images found in: {sample_dir}")

    robot_to_base_rotations = []
    robot_to_base_translations = []
    target_to_camera_rotations = []
    target_to_camera_translations = []
    sample_summaries: list[SampleSummary] = []
    gripper_to_target_transforms = []

    for image_path in image_paths:
        pose_path = image_path.with_suffix(".json")
        if not pose_path.exists():
            print(f"Skipping sample without pose file: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        found, corners, _ = find_chessboard_corners(image, args.cols, args.rows)
        if not found or corners is None:
            print(f"Skipping sample without detected corners: {image_path}")
            continue

        robot_pose = RobotPose.from_dict(load_json(pose_path))
        rvec, tvec, pnp_error_px = solve_pnp(object_points, corners, camera_matrix, dist_coeffs)

        t_base_gripper = robot_pose.to_transform()
        t_gripper_base = invert_transform(t_base_gripper)
        t_camera_target = rvec_tvec_to_transform(rvec, tvec)

        r_g2b, t_g2b = split_transform(t_gripper_base)
        r_t2c, t_t2c = split_transform(t_camera_target)

        robot_to_base_rotations.append(r_g2b)
        robot_to_base_translations.append(t_g2b.reshape(3, 1))
        target_to_camera_rotations.append(r_t2c)
        target_to_camera_translations.append(t_t2c.reshape(3, 1))
        sample_summaries.append(
            SampleSummary(
                image_path=str(image_path),
                pose_path=str(pose_path),
                pnp_error_px=pnp_error_px,
            )
        )

    if len(sample_summaries) < 8:
        raise RuntimeError(f"Not enough valid samples. Need at least 8, got {len(sample_summaries)}.")

    method_flag = getattr(cv2, HAND_EYE_METHODS[args.method])

    # Eye-to-hand: invert base->gripper into gripper->base before calling OpenCV.
    r_base_camera, t_base_camera = cv2.calibrateHandEye(
        robot_to_base_rotations,
        robot_to_base_translations,
        target_to_camera_rotations,
        target_to_camera_translations,
        method=method_flag,
    )

    t_base_camera = np.asarray(t_base_camera, dtype=np.float64).reshape(3)
    transform_base_camera = np.eye(4, dtype=np.float64)
    transform_base_camera[:3, :3] = np.asarray(r_base_camera, dtype=np.float64)
    transform_base_camera[:3, 3] = t_base_camera
    transform_camera_base = invert_transform(transform_base_camera)

    for idx, _summary in enumerate(sample_summaries):
        t_gripper_base = np.eye(4, dtype=np.float64)
        t_gripper_base[:3, :3] = robot_to_base_rotations[idx]
        t_gripper_base[:3, 3] = np.asarray(robot_to_base_translations[idx]).reshape(3)

        t_camera_target = np.eye(4, dtype=np.float64)
        t_camera_target[:3, :3] = target_to_camera_rotations[idx]
        t_camera_target[:3, 3] = np.asarray(target_to_camera_translations[idx]).reshape(3)

        gripper_to_target = t_gripper_base @ transform_base_camera @ t_camera_target
        gripper_to_target_transforms.append(gripper_to_target)

    gt_translations = np.asarray([transform[:3, 3] for transform in gripper_to_target_transforms])
    gt_mean = gt_translations.mean(axis=0)
    gt_std = gt_translations.std(axis=0)

    output_path = Path(args.output)
    payload = {
        "calibration_type": "eye_to_hand",
        "camera_model": "orbbec_color",
        "hand_eye_method": args.method,
        "board": {
            "cols": args.cols,
            "rows": args.rows,
            "square_size_mm": args.square_size_mm,
        },
        "samples_used": len(sample_summaries),
        "intrinsics_path": str(Path(args.intrinsics).resolve()),
        "base_to_camera": transform_to_dict(transform_base_camera),
        "camera_to_base": transform_to_dict(transform_camera_base),
        "gripper_to_target_translation_mean_m": gt_mean.tolist(),
        "gripper_to_target_translation_std_m": gt_std.tolist(),
        "sample_details": [summary.to_dict() for summary in sample_summaries],
    }
    save_json(output_path, payload)

    print(f"Hand-eye calibration finished with {len(sample_summaries)} samples.")
    print(f"base -> camera translation (mm): {(transform_base_camera[:3, 3] * 1000.0).round(3).tolist()}")
    print(f"Result written to: {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "capture":
        return run_capture(args)
    if args.command == "solve":
        return run_solve(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
