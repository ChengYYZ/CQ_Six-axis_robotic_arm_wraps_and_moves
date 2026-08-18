from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chessboard_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid
    object_points *= square_size_mm / 1000.0
    return object_points


def rpy_xyz_to_matrix(rpy_rad: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy_rad
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_rpy_xyz(rotation: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def make_transform(rotation: np.ndarray, translation_m: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def split_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, :3].copy(), transform[:3, 3].copy()


def transform_to_dict(transform: np.ndarray) -> dict:
    rotation, translation_m = split_transform(transform)
    return {
        "rotation_matrix": rotation.tolist(),
        "translation_m": translation_m.tolist(),
        "translation_mm": (translation_m * 1000.0).tolist(),
        "rpy_rad_xyz": matrix_to_rpy_xyz(rotation).tolist(),
    }


def find_chessboard_corners(
    bgr_image: np.ndarray,
    cols: int,
    rows: int,
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    import cv2

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, None)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return bool(found), corners, gray


def draw_chessboard_overlay(
    bgr_image: np.ndarray,
    cols: int,
    rows: int,
    corners: np.ndarray | None,
    found: bool,
) -> np.ndarray:
    import cv2

    canvas = bgr_image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(canvas, (cols, rows), corners, found)
    return canvas


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    import cv2

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed for the current sample.")

    reprojected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    error = np.linalg.norm(reprojected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    mean_error = float(error.mean())
    return rvec, tvec, mean_error


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    return make_transform(rotation, translation)


@dataclass
class SampleSummary:
    image_path: str
    pose_path: str
    pnp_error_px: float

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "pose_path": self.pose_path,
            "pnp_error_px": self.pnp_error_px,
        }

