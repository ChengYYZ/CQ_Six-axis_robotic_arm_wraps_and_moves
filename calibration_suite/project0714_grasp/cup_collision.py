"""Observed point intrusion into a suction body; millimetres throughout.

An empty hit set is NOT proof of free space: unseen surfaces are not represented.
The origin is the contact-face centre. X=79.6, Y=59.6, body depth=43.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CupIntrusion:
    valid_points: int
    hit_indices: np.ndarray
    max_depth_mm: float

    @property
    def observed_collision(self) -> bool:
        return bool(len(self.hit_indices))


def check_cup_volume(points_base_mm, center_base_mm, rotation_base_from_cup,
                     body_z_sign=1, contact_tolerance_mm=2.0) -> CupIntrusion:
    """Exclude only the contact-side tolerance slab, never the entire parcel.

    body_z_sign=+1 means the mounting end is along cup +Z; -1 means -Z.
    Returned indices refer to the original cloud, including when it has NaNs.
    """
    points = np.asarray(points_base_mm, dtype=float)
    center = np.asarray(center_base_mm, dtype=float)
    rotation = np.asarray(rotation_base_from_cup, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3)")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Contact centre must be a finite 3-vector")
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError("Cup rotation must be a proper rotation matrix")
    if body_z_sign not in (-1, 1):
        raise ValueError("body_z_sign must be -1 or +1")
    if not np.isfinite(contact_tolerance_mm) or not 0 <= contact_tolerance_mm < 43:
        raise ValueError("Contact tolerance must be in [0, 43) mm")
    indices = np.flatnonzero(np.all(np.isfinite(points), axis=1))
    local = (points[indices] - center) @ rotation
    depth = local[:, 2] * body_z_sign
    hits = ((np.abs(local[:, 0]) <= 39.8) & (np.abs(local[:, 1]) <= 29.8)
            & (depth > contact_tolerance_mm) & (depth <= 43.0))
    return CupIntrusion(len(indices), indices[hits], float(depth[hits].max()) if hits.any() else 0.)
