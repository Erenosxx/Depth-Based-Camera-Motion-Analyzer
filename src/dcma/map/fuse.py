"""Derinlik + RGB + T_wc → dünya nokta bulutu; voxel seyreltme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcma.calib.intrinsics import Intrinsics

DEFAULT_STRIDE = 4


@dataclass
class FusedCloud:
    xyz: np.ndarray  # (N, 3) float64 dünya
    rgb: np.ndarray  # (N, 3) uint8


def unproject_frame(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: Intrinsics,
    T_wc: np.ndarray,
    stride: int = DEFAULT_STRIDE,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    rgb = np.asarray(rgb)
    h, w = depth.shape[:2]
    vs = np.arange(0, h, int(stride))
    us = np.arange(0, w, int(stride))
    uu, vv = np.meshgrid(us, vs)
    u = uu.ravel().astype(np.float64)
    v = vv.ravel().astype(np.float64)
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    z = depth[vi, ui].astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    valid = (
        np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
        & (z > min_depth) & (z < max_depth)
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    pts = np.stack([x[valid], y[valid], z[valid], np.ones(int(valid.sum()))])
    world = (np.asarray(T_wc, dtype=np.float64) @ pts)[:3].T
    color = rgb[vi[valid], ui[valid]].astype(np.uint8)
    return world, color


def voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if len(xyz) == 0:
        return xyz.reshape(0, 3), rgb.reshape(0, 3)
    if voxel <= 0:
        raise ValueError(f"voxel pozitif olmalı: {voxel}")
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], rgb[idx]


def fuse_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    K: Intrinsics,
    stride: int = DEFAULT_STRIDE,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
    voxel: float = 0.03,
) -> FusedCloud:
    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    for depth, rgb, T_wc in frames:
        xyz, color = unproject_frame(
            depth, rgb, K, T_wc, stride=stride,
            min_depth=min_depth, max_depth=max_depth)
        if len(xyz):
            chunks_xyz.append(xyz)
            chunks_rgb.append(color)
    if not chunks_xyz:
        return FusedCloud(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8))
    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    return FusedCloud(xyz=xyz, rgb=rgb)
