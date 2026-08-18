from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from project0714_calib.common import load_json, save_json


@dataclass
class PlaneModel:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_indices: np.ndarray


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


@dataclass
class SupportPlaneAssignment:
    region_id_map: np.ndarray
    distance_mm_map: np.ndarray
    support_index_map: np.ndarray


@dataclass
class SupportPlaneModel:
    overall_mask: np.ndarray
    valid_mask: np.ndarray
    support_masks: dict[str, np.ndarray]
    support_planes: dict[str, PlaneModel]
    assignment: SupportPlaneAssignment
    notes: list[str]


SUPPORT_REGION_SPECS: tuple[SupportRegionSpec, ...] = (
    SupportRegionSpec("floor", "1", "Floor", (255, 200, 0)),
    SupportRegionSpec("left_wall", "2", "Left Wall", (255, 120, 0)),
    SupportRegionSpec("right_wall", "3", "Right Wall", (0, 220, 255)),
    SupportRegionSpec("back_wall", "4", "Back Wall", (180, 80, 255)),
)


def polygon_from_payload(payload: object) -> np.ndarray | None:
    points = np.asarray(payload if payload is not None else [], dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        return None
    return points


def load_roi_config(path: str | Path) -> RoiConfig:
    roi_path = Path(path)
    if not roi_path.exists():
        return RoiConfig(overall_polygon=None, support_polygons={})

    payload = load_json(roi_path)
    if "polygon_pixels" in payload:
        overall_polygon = polygon_from_payload(payload.get("polygon_pixels"))
        return RoiConfig(overall_polygon=overall_polygon, support_polygons={})

    overall_polygon = polygon_from_payload(payload.get("overall_polygon_pixels"))
    support_polygons: dict[str, np.ndarray] = {}
    support_block = payload.get("support_polygons", {})
    if isinstance(support_block, dict):
        for spec in SUPPORT_REGION_SPECS:
            polygon = polygon_from_payload(support_block.get(spec.region_id))
            if polygon is not None:
                support_polygons[spec.region_id] = polygon
    return RoiConfig(overall_polygon=overall_polygon, support_polygons=support_polygons)


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
    }
    save_json(path, payload)


def delete_roi_config(path: str | Path) -> None:
    roi_path = Path(path)
    if roi_path.exists():
        roi_path.unlink()


def get_support_region_spec(region_id: str) -> SupportRegionSpec | None:
    for spec in SUPPORT_REGION_SPECS:
        if spec.region_id == region_id:
            return spec
    return None


def roi_mask_from_polygon(image_shape: tuple[int, int], polygon: np.ndarray | None) -> np.ndarray | None:
    if polygon is None or len(polygon) < 3:
        return None

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask > 0


def polygon_from_mask(mask: np.ndarray) -> np.ndarray | None:
    points_yx = np.column_stack(np.nonzero(mask))
    if len(points_yx) < 3:
        return None
    points_xy = np.column_stack([points_yx[:, 1], points_yx[:, 0]]).astype(np.int32)
    return cv2.convexHull(points_xy).reshape(-1, 2)


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


def fit_plane_svd(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points_mm.mean(axis=0)
    centered = points_mm - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def fit_plane_ransac(points_mm: np.ndarray, threshold_mm: float, iterations: int) -> PlaneModel:
    if len(points_mm) < 3:
        raise RuntimeError("At least 3 points are required to fit a plane.")

    rng = np.random.default_rng(523)
    best_inliers = np.empty((0,), dtype=np.int32)
    best_normal = None

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

    if best_normal is None or len(best_inliers) < 3:
        raise RuntimeError("Failed to fit a dominant plane.")

    centroid, normal = fit_plane_svd(points_mm[best_inliers])
    if float(normal @ centroid) > 0.0:
        normal = -normal
    d = float(normal @ centroid)
    return PlaneModel(normal=normal, d=d, centroid=centroid, inlier_indices=best_inliers)


def plane_signed_distance(points_mm: np.ndarray, plane: PlaneModel) -> np.ndarray:
    return points_mm @ plane.normal - plane.d


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


def build_support_plane_model(
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    roi_config: RoiConfig,
    min_depth_mm: float,
    max_depth_mm: float,
    plane_threshold_mm: float,
    plane_iterations: int,
    tray_erode_px: int,
) -> SupportPlaneModel:
    valid_mask = (depth_mm > min_depth_mm) & (depth_mm < max_depth_mm)
    if not np.any(valid_mask):
        raise RuntimeError("No valid depth pixels are available.")

    grid_y, grid_x = np.indices(depth_mm.shape, dtype=np.int32)
    all_pixels = np.column_stack([grid_x[valid_mask], grid_y[valid_mask]])
    all_depth = depth_mm[valid_mask]
    all_points = pixel_to_camera_points(all_pixels, all_depth, camera_matrix)
    fallback_plane = fit_plane_ransac(all_points, plane_threshold_mm, plane_iterations)
    _, fallback_height_map = dense_height_map(depth_mm, camera_matrix, fallback_plane, min_depth_mm, max_depth_mm)

    overall_mask = roi_mask_from_polygon(depth_mm.shape, roi_config.overall_polygon)
    if overall_mask is None:
        overall_mask = build_tray_mask(valid_mask, fallback_height_map, plane_threshold_mm, tray_erode_px)
        if not np.any(overall_mask):
            overall_mask = valid_mask.copy()
    else:
        overall_mask = overall_mask & valid_mask

    if not np.any(overall_mask):
        raise RuntimeError("Overall ROI is empty after applying valid depth.")

    support_masks: dict[str, np.ndarray] = {}
    support_planes: dict[str, PlaneModel] = {}
    notes: list[str] = [f"overall_pixels={int(np.count_nonzero(overall_mask))}"]

    for spec in SUPPORT_REGION_SPECS:
        polygon = roi_config.support_polygons.get(spec.region_id)
        support_mask = roi_mask_from_polygon(depth_mm.shape, polygon)
        if support_mask is None:
            continue
        support_mask = support_mask & overall_mask & valid_mask
        support_pixels = int(np.count_nonzero(support_mask))
        notes.append(f"{spec.region_id}:pixels={support_pixels}")
        if support_pixels < 80:
            continue

        ys, xs = np.nonzero(support_mask)
        support_pixels_xy = np.column_stack([xs, ys]).astype(np.int32)
        support_depth = depth_mm[support_mask]
        support_points = pixel_to_camera_points(support_pixels_xy, support_depth, camera_matrix)
        try:
            plane = fit_plane_ransac(support_points, plane_threshold_mm, plane_iterations)
        except Exception:
            continue
        support_masks[spec.region_id] = support_mask
        support_planes[spec.region_id] = plane

    if not support_planes:
        support_masks["overall"] = overall_mask
        support_planes["overall"] = fallback_plane
        notes.append("support_planes=fallback_overall")

    distance_stack: list[np.ndarray] = []
    region_ids: list[str] = []
    for region_id, plane in support_planes.items():
        _, height_map = dense_height_map(depth_mm, camera_matrix, plane, min_depth_mm, max_depth_mm)
        distance = np.full(depth_mm.shape, np.inf, dtype=np.float32)
        mask = support_masks[region_id] & valid_mask
        distance[mask] = np.abs(height_map[mask])
        distance_stack.append(distance)
        region_ids.append(region_id)

    distances = np.stack(distance_stack, axis=0)
    support_index_map = np.argmin(distances, axis=0).astype(np.int32)
    distance_mm_map = np.take_along_axis(distances, support_index_map[None, :, :], axis=0)[0]
    region_id_map = np.full(depth_mm.shape, "", dtype="<U16")
    for index, region_id in enumerate(region_ids):
        region_id_map[support_index_map == index] = region_id

    assignment = SupportPlaneAssignment(
        region_id_map=region_id_map,
        distance_mm_map=distance_mm_map,
        support_index_map=support_index_map,
    )
    return SupportPlaneModel(
        overall_mask=overall_mask,
        valid_mask=valid_mask,
        support_masks=support_masks,
        support_planes=support_planes,
        assignment=assignment,
        notes=notes,
    )

