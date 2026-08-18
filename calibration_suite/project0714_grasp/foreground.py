from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .support_planes import SupportPlaneModel, dense_height_map


@dataclass
class ForegroundModel:
    nearest_region_id_map: np.ndarray
    nearest_height_mm_map: np.ndarray
    foreground_mask: np.ndarray
    foreground_pixels: int
    notes: list[str]


def clean_mask(mask: np.ndarray, close_px: int, open_px: int) -> np.ndarray:
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


def build_foreground_from_support_planes(
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    support_model: SupportPlaneModel,
    min_depth_mm: float,
    max_depth_mm: float,
    object_height_mm: float,
    mask_close_px: int,
    mask_open_px: int,
) -> ForegroundModel:
    region_ids = sorted(support_model.support_planes.keys())
    if not region_ids:
        raise RuntimeError("No support planes are available.")

    height_maps: list[np.ndarray] = []
    abs_distance_maps: list[np.ndarray] = []
    for region_id in region_ids:
        plane = support_model.support_planes[region_id]
        _, height_map = dense_height_map(depth_mm, camera_matrix, plane, min_depth_mm, max_depth_mm)
        height_maps.append(height_map)
        abs_map = np.full(depth_mm.shape, np.inf, dtype=np.float32)
        local_mask = support_model.support_masks[region_id] & support_model.valid_mask
        abs_map[local_mask] = np.abs(height_map[local_mask])
        abs_distance_maps.append(abs_map)

    distance_stack = np.stack(abs_distance_maps, axis=0)
    nearest_index = np.argmin(distance_stack, axis=0).astype(np.int32)
    nearest_height_mm_map = np.full(depth_mm.shape, np.nan, dtype=np.float32)
    nearest_region_id_map = np.full(depth_mm.shape, "", dtype="<U16")

    for index, region_id in enumerate(region_ids):
        region_mask = nearest_index == index
        nearest_height_mm_map[region_mask] = height_maps[index][region_mask]
        nearest_region_id_map[region_mask] = region_id

    raw_foreground_mask = (
        support_model.overall_mask
        & support_model.valid_mask
        & np.isfinite(nearest_height_mm_map)
        & (nearest_height_mm_map > object_height_mm)
    )
    foreground_mask = clean_mask(raw_foreground_mask, mask_close_px, mask_open_px)
    foreground_pixels = int(np.count_nonzero(foreground_mask))
    notes = [
        f"foreground_pixels={foreground_pixels}",
        f"nearest_regions={len(region_ids)}",
    ]
    return ForegroundModel(
        nearest_region_id_map=nearest_region_id_map,
        nearest_height_mm_map=nearest_height_mm_map,
        foreground_mask=foreground_mask,
        foreground_pixels=foreground_pixels,
        notes=notes,
    )

