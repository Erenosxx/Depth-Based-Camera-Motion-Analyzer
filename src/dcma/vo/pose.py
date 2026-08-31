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
# ~8°/0.15 s üstü bilinçli pivot; yürüyerek viraj genelde bunun altında kalır.
YAW_INPLACE_DEG = 8.0
# PnP yaw ofis L-köşesinde ~%10–15 şişiyor; 1.0 ham.
DEFAULT_YAW_SCALE = 0.88
# |sağ| > |ileri|: duvara bakıp kayınca PnP'nin uydurduğu ileri bileşen.
STRAFE_RIGHT_OVER_FORWARD = 1.0


def yaw_deg(R: np.ndarray) -> float:
    """Kamera yaw (derece). OpenCV Rodrigues +Y ile aynı işaret; negatif = sağa bakış."""
    R = np.asarray(R, dtype=np.float64)
    return float(np.rad2deg(np.arctan2(R[0, 2], R[2, 2])))


def scale_yaw(R: np.ndarray, scale: float) -> np.ndarray:
    """Yaw bileşenini `scale` ile küçült/büyüt; pitch/roll aynı kalır."""
    R = np.asarray(R, dtype=np.float64)
    yaw = np.arctan2(R[0, 2], R[2, 2])
    delta = yaw * (float(scale) - 1.0)
    R_delta, _ = cv2.Rodrigues(np.array([0.0, delta, 0.0], dtype=np.float64))
    return R_delta @ R


def maybe_inplace_yaw(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Büyük yaw adımında ötelemeyi at: derinlik yanlılığı pivotu yürüyüş yapmasın."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel().copy()
    if abs(yaw_deg(R)) >= YAW_INPLACE_DEG:
        return R, np.zeros(3, dtype=np.float64)
    return R, t


def maybe_lateral_strafe(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Küçük yaw + yan öteleme: sahte ileri bileşeni sıfırla (duvar ayağın dibine basmasın)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel().copy()
    if abs(yaw_deg(R)) >= YAW_INPLACE_DEG:
        return R, t
    C = -R.T @ t
    if abs(C[0]) > STRAFE_RIGHT_OVER_FORWARD * abs(C[2]) and abs(C[0]) > 1e-9:
        C = C.copy()
        C[2] = 0.0
        t = -R @ C
    return R, t


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
