"""Kuş bakışı occupancy: derinlik + T_wc → dünya xz (sağa, ileri)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dcma.calib.intrinsics import Intrinsics

DEFAULT_RES = 0.10
DEFAULT_STRIDE = 12


class OccupancyGrid:
    """Seyrek hit listesi; hücreler ızgara indeksi (ix, iz) olarak tutulur."""

    def __init__(self, resolution: float = DEFAULT_RES, stride: int = DEFAULT_STRIDE,
                 y_lo: float = -0.5, y_hi: float = 0.5,
                 min_depth: float = 0.3, max_depth: float = 15.0,
                 near_m: float = 0.8, z_rel: float = 0.55,
                 center_frac: float = 0.22) -> None:
        if resolution <= 0:
            raise ValueError(f"çözünürlük pozitif olmalı: {resolution}")
        self.resolution = float(resolution)
        self.stride = int(stride)
        self.y_lo = float(y_lo)
        self.y_hi = float(y_hi)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.near_m = float(near_m)
        self.z_rel = float(z_rel)
        self.center_frac = float(center_frac)
        self._frames: list[int] = []
        self._ix: list[int] = []
        self._iz: list[int] = []

    def splat(self, depth: np.ndarray, K: Intrinsics, T_wc: np.ndarray,
              frame_idx: int) -> int:
        """Derinlik karesini dünya xz'ye basar. Eklenen hit sayısı."""
        depth = np.asarray(depth)
        h, w = depth.shape[:2]
        v0, v1 = int(h * 0.30), max(int(h * 0.70), int(h * 0.30) + 1)
        vs = np.arange(v0, v1, self.stride)
        us = np.arange(0, w, self.stride)
        if len(vs) == 0 or len(us) == 0:
            return 0
        uu, vv = np.meshgrid(us, vs)
        u = uu.ravel().astype(np.float64)
        v = vv.ravel().astype(np.float64)
        ui = np.clip(np.round(u).astype(int), 0, w - 1)
        vi = np.clip(np.round(v).astype(int), 0, h - 1)
        z = depth[vi, ui].astype(np.float64)
        x = (u - K.cx) * z / K.fx
        y = (v - K.cy) * z / K.fy
        # Karşı duvar (görüntü ortası) yana kayınca yolun üstüne basılmasın;
        # koridor yan duvarları kenarda kalır.
        off_center = np.abs(u - K.cx) >= self.center_frac * w
        valid = (
            np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
            & (z > self.min_depth) & (z < self.max_depth)
            & (y >= self.y_lo) & (y <= self.y_hi)
            & off_center
        )
        if self.z_rel > 0 and np.any(valid):
            med = float(np.median(z[valid]))
            valid = valid & (z >= self.z_rel * med)
        if not np.any(valid):
            return 0
        pts = np.stack([x[valid], y[valid], z[valid], np.ones(int(valid.sum()))])
        T_wc = np.asarray(T_wc, dtype=np.float64)
        world = T_wc @ pts
        cam = T_wc[:3, 3]
        dist_xz = np.hypot(world[0] - cam[0], world[2] - cam[2])
        keep = dist_xz >= self.near_m
        if not np.any(keep):
            return 0
        world = world[:, keep]
        ix = np.floor(world[0] / self.resolution).astype(int)
        iz = np.floor(world[2] / self.resolution).astype(int)
        n = int(ix.size)
        self._frames.extend([int(frame_idx)] * n)
        self._ix.extend(ix.tolist())
        self._iz.extend(iz.tolist())
        return n

    def cells_upto(self, frame_idx: int) -> np.ndarray:
        """frame_idx'e kadar (dahil) dolu hücre merkezleri, şekil (N, 2)=(sağa, ileri)."""
        if not self._frames:
            return np.zeros((0, 2), dtype=np.float64)
        frames = np.asarray(self._frames, dtype=np.int64)
        mask = frames <= int(frame_idx)
        if not np.any(mask):
            return np.zeros((0, 2), dtype=np.float64)
        pairs = np.stack([
            np.asarray(self._ix, dtype=np.int64)[mask],
            np.asarray(self._iz, dtype=np.int64)[mask],
        ], axis=1)
        uniq = np.unique(pairs, axis=0)
        return (uniq.astype(np.float64) + 0.5) * self.resolution

    def dense(self, frame_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(counts[iz, ix], origin_xz) — origin sol-alt hücre köşesi (sağa, ileri)."""
        if frame_idx is None:
            frame_idx = max(self._frames) if self._frames else -1
        cells = self.cells_upto(frame_idx)
        if len(cells) == 0:
            return np.zeros((1, 1), dtype=np.float32), np.zeros(2, dtype=np.float64)
        ix = np.floor(cells[:, 0] / self.resolution).astype(int)
        iz = np.floor(cells[:, 1] / self.resolution).astype(int)
        ix0, iz0 = int(ix.min()), int(iz.min())
        counts = np.zeros((int(iz.max() - iz0) + 1, int(ix.max() - ix0) + 1),
                          dtype=np.float32)
        counts[iz - iz0, ix - ix0] = 1.0
        origin = np.array([ix0 * self.resolution, iz0 * self.resolution], dtype=np.float64)
        return counts, origin

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            resolution=np.float64(self.resolution),
            stride=np.int32(self.stride),
            y_lo=np.float64(self.y_lo),
            y_hi=np.float64(self.y_hi),
            min_depth=np.float64(self.min_depth),
            max_depth=np.float64(self.max_depth),
            near_m=np.float64(self.near_m),
            z_rel=np.float64(self.z_rel),
            center_frac=np.float64(self.center_frac),
            frames=np.asarray(self._frames, dtype=np.int32),
            ix=np.asarray(self._ix, dtype=np.int32),
            iz=np.asarray(self._iz, dtype=np.int32),
        )
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> "OccupancyGrid":
        data = np.load(path)
        grid = cls(
            resolution=float(data["resolution"]),
            stride=int(data["stride"]) if "stride" in data.files else DEFAULT_STRIDE,
            y_lo=float(data["y_lo"]) if "y_lo" in data.files else -0.5,
            y_hi=float(data["y_hi"]) if "y_hi" in data.files else 0.5,
            min_depth=float(data["min_depth"]) if "min_depth" in data.files else 0.3,
            max_depth=float(data["max_depth"]) if "max_depth" in data.files else 15.0,
            near_m=float(data["near_m"]) if "near_m" in data.files else 0.8,
            z_rel=float(data["z_rel"]) if "z_rel" in data.files else 0.55,
            center_frac=float(data["center_frac"]) if "center_frac" in data.files else 0.22,
        )
        grid._frames = np.asarray(data["frames"], dtype=int).tolist()
        grid._ix = np.asarray(data["ix"], dtype=int).tolist()
        grid._iz = np.asarray(data["iz"], dtype=int).tolist()
        return grid

    def write_png(self, path: str | Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        counts, origin = self.dense()
        res = self.resolution
        h, w = counts.shape
        extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(counts, origin="lower", extent=extent, cmap="gray_r",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel("sağa (m)")
        ax.set_ylabel("ileri (m)")
        ax.set_title("Occupancy (kabaca duvar)")
        ax.set_aspect("equal")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
