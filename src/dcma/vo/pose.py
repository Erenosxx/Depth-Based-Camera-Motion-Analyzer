"""Kare çiftleri arası poz kestirimi.

Konvansiyonlar (OpenCV kamera çerçevesi):
  +x saga, +y ASAGI, +z ileri
  solvePnP cikti: X_cam2 = R @ X_cam1 + t
  kamera-2 merkezinin kare-1 koordinatindaki yeri: C = -R.T @ t
  rapor eksenleri: ileri = C[2], saga = C[0], yukari = -C[1]
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dcma.calib.intrinsics import Intrinsics

MIN_POINTS = 6


@dataclass
class PoseResult:
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    inliers: int
    reproj_err: float

    @property
    def forward(self) -> float:
        return float(self.C[2])

    @property
    def right(self) -> float:
        return float(self.C[0])

    @property
    def up(self) -> float:
        return float(-self.C[1])

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.C))


def backproject(depth: np.ndarray, K: Intrinsics,
                pts2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2B noktaları derinlik ve K ile kamera koordinatlarında 3B'ye taşır."""
    u = np.asarray(pts2d, dtype=np.float64)[:, 0]
    v = np.asarray(pts2d, dtype=np.float64)[:, 1]
    h, w = depth.shape[:2]
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    z = depth[vi, ui].astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=1), z


def estimate_pose(P3d: np.ndarray, p2: np.ndarray, K: Intrinsics,
                  reproj_thresh: float = 2.0, iterations: int = 2000,
                  confidence: float = 0.999) -> PoseResult | None:
    """kare-1'in 3B noktaları ile kare-2'deki izdüşümlerinden pozu çözer."""
    obj = np.ascontiguousarray(P3d, dtype=np.float64)
    img = np.ascontiguousarray(p2, dtype=np.float64)
    if len(obj) < MIN_POINTS or len(obj) != len(img):
        return None

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K.matrix, None,
        reprojectionError=reproj_thresh,
        iterationsCount=iterations,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < MIN_POINTS:
        return None

    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K.matrix, None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).ravel()

    proj, _ = cv2.projectPoints(obj[idx], rvec, tvec, K.matrix, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1).mean())

    return PoseResult(R=R, t=t, C=(-R.T @ t), inliers=int(len(idx)), reproj_err=err)
