from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project0714_calib.common import (
    chessboard_object_points,
    draw_chessboard_overlay,
    ensure_dir,
    find_chessboard_corners,
    save_json,
    timestamp_tag,
)
from project0714_calib.orbbec_camera import CameraOpenOptions, OrbbecColorCamera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project0714 Orbbec 相机内参标定")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="采集棋盘格图像")
    add_board_args(capture)
    add_camera_args(capture)
    capture.add_argument(
        "--save-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "intrinsics" / "images"),
        help="图像保存目录",
    )

    calibrate = subparsers.add_parser("calibrate", help="根据图像计算相机内参")
    add_board_args(calibrate)
    calibrate.add_argument(
        "--image-dir",
        default=str(Path(__file__).resolve().parent / "workspace" / "intrinsics" / "images"),
        help="采集图像目录",
    )
    calibrate.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "workspace" / "intrinsics" / "camera_intrinsics.json"),
        help="输出 JSON 路径",
    )
    return parser


def add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cols", type=int, required=True, help="棋盘格内角点列数")
    parser.add_argument("--rows", type=int, required=True, help="棋盘格内角点行数")
    parser.add_argument("--square-size-mm", type=float, required=True, help="棋盘格方格边长，单位 mm")


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=1280, help="相机宽度")
    parser.add_argument("--height", type=int, default=800, help="相机高度")
    parser.add_argument("--fps", type=int, default=10, help="相机帧率")


def run_capture(args: argparse.Namespace) -> int:
    import cv2

    save_dir = ensure_dir(Path(args.save_dir))
    camera = OrbbecColorCamera(CameraOpenOptions(args.width, args.height, args.fps))
    captured = 0

    try:
        camera.start()
        while True:
            frame = camera.get_bgr_frame(timeout_ms=200)
            if frame is None:
                continue

            found, corners, _ = find_chessboard_corners(frame, args.cols, args.rows)
            overlay = draw_chessboard_overlay(frame, args.cols, args.rows, corners, found)
            status = f"saved={captured}  detected={'YES' if found else 'NO'}  key:s save  q quit"
            cv2.putText(
                overlay,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 120, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Project0714 Intrinsic Capture", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                if not found:
                    print("当前帧未检测到完整棋盘格，已跳过。")
                    continue
                image_path = save_dir / f"intrinsic_{timestamp_tag()}.png"
                cv2.imwrite(str(image_path), frame)
                captured += 1
                print(f"[{captured}] 已保存: {image_path}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    import cv2

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"未找到标定图像: {image_dir}")

    board_object_points = chessboard_object_points(args.cols, args.rows, args.square_size_mm)
    object_points = []
    image_points = []
    image_size = None
    used_images: list[str] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"跳过无法读取的图像: {image_path}")
            continue
        found, corners, gray = find_chessboard_corners(image, args.cols, args.rows)
        if not found or corners is None:
            print(f"跳过未检测到角点的图像: {image_path}")
            continue

        object_points.append(board_object_points.astype(np.float32))
        image_points.append(corners.astype(np.float32))
        image_size = (gray.shape[1], gray.shape[0])
        used_images.append(str(image_path))

    if len(object_points) < 8:
        raise RuntimeError(f"有效图像不足，至少需要 8 张，当前仅 {len(object_points)} 张。")

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    total_error = 0.0
    total_points = 0
    per_image_errors = []
    for idx, objp in enumerate(object_points):
        projected, _ = cv2.projectPoints(objp, rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs)
        diff = projected.reshape(-1, 2) - image_points[idx].reshape(-1, 2)
        err = np.linalg.norm(diff, axis=1)
        total_error += float(err.sum())
        total_points += int(err.size)
        per_image_errors.append(float(err.mean()))

    mean_error = total_error / max(total_points, 1)
    output_path = Path(args.output)
    payload = {
        "camera_model": "orbbec_color",
        "board": {
            "cols": args.cols,
            "rows": args.rows,
            "square_size_mm": args.square_size_mm,
        },
        "image_size": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "mean_reprojection_error": mean_error,
        "per_image_mean_error": per_image_errors,
        "used_images": used_images,
    }
    save_json(output_path, payload)

    print(f"标定完成，使用图像 {len(used_images)} 张。")
    print(f"平均重投影误差: {mean_error:.4f} px")
    print(f"结果已写入: {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "capture":
        return run_capture(args)
    if args.command == "calibrate":
        return run_calibrate(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

