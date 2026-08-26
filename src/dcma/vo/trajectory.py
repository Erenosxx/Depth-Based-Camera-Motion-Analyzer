"""Poz birikimi ve metre cinsinden özetler.

Poz konvansiyonu: T_wc[k] = dunya-kameradan donusum, dunya = ilk keyframe.
Adim (R, t) su anlama gelir: X_cam(k+1) = R @ X_cam(k) + t
Dolayisiyla T_wc[k+1] = T_wc[k] @ inv([R|t]).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def yaw_deg_from_R(R: np.ndarray) -> float:
    """Kamera bakış yaw'ı (derece). Negatif = sağa dönüş (üstten bakışta saat yönü)."""
    R = np.asarray(R, dtype=np.float64)
    return float(np.rad2deg(np.arctan2(-R[0, 2], R[2, 2])))


class Trajectory:
    def __init__(self) -> None:
        self.poses: list[np.ndarray] = [np.eye(4)]
        self._steps: list[dict[str, Any]] = []

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def positions(self) -> np.ndarray:
        return np.array([T[:3, 3] for T in self.poses])

    @property
    def net_displacement(self) -> np.ndarray:
        return self.positions[-1] - self.positions[0]

    @property
    def path_length(self) -> float:
        return float(sum(s["distance"] for s in self._steps))

    @property
    def odometer(self) -> dict[str, float]:
        return {
            "forward": float(sum(s["forward"] for s in self._steps)),
            "right": float(sum(s["right"] for s in self._steps)),
            "up": float(sum(s["up"] for s in self._steps)),
        }

    def add_step(self, R: np.ndarray, t: np.ndarray,
                 inliers: int | None = None,
                 reproj_err: float | None = None,
                 frame_from: int | None = None,
                 frame_to: int | None = None) -> None:
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()

        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        self.poses.append(self.poses[-1] @ np.linalg.inv(dT))

        C = -R.T @ t
        yaw0 = yaw_deg_from_R(self.poses[-2][:3, :3])
        yaw1 = yaw_deg_from_R(self.poses[-1][:3, :3])
        yaw_delta = float(np.rad2deg(np.arctan2(
            np.sin(np.deg2rad(yaw1 - yaw0)),
            np.cos(np.deg2rad(yaw1 - yaw0)))))
        self._steps.append({
            "frame_from": frame_from,
            "frame_to": frame_to,
            "forward": float(C[2]),
            "right": float(C[0]),
            "up": float(-C[1]),
            "yaw_deg": yaw_delta,
            "distance": float(np.linalg.norm(C)),
            "inliers": inliers,
            "reproj_err": reproj_err,
        })

    def to_dict(self) -> dict[str, Any]:
        net = self.net_displacement
        return {
            "step_count": self.step_count,
            "odometer": self.odometer,
            "net_displacement": {
                "forward": float(net[2]),
                "right": float(net[0]),
                "up": float(-net[1]),
                "magnitude": float(np.linalg.norm(net)),
            },
            "path_length": self.path_length,
            "positions": self.positions.tolist(),
            "poses": [T.tolist() for T in self.poses],
            "steps": self._steps,
        }
