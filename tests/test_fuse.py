"""Sentetik koridor: duvarlar dünya x=±1 m; ikinci kamera +2 m ileri."""

from __future__ import annotations

import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.map.fuse import unproject_frame, voxel_downsample, fuse_frames


def _corridor_depth(K: Intrinsics, wall_x: float = 1.0) -> np.ndarray:
    depth = np.full((K.height, K.width), np.nan, dtype=np.float32)
    us = np.arange(K.width, dtype=np.float64)
    left, right = us < K.cx, us > K.cx
    depth[:, left] = ((-wall_x) * K.fx / (us[left] - K.cx))[np.newaxis, :]
    depth[:, right] = (wall_x * K.fx / (us[right] - K.cx))[np.newaxis, :]
    return depth


def _gray_rgb(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_identity_unproject_hits_walls_at_plus_minus_one_metre():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    xyz, rgb = unproject_frame(
        _corridor_depth(K), _gray_rgb(K.height, K.width), K, np.eye(4),
        stride=4, min_depth=0.3, max_depth=15.0)
    assert xyz.shape[1] == 3 and rgb.shape == (len(xyz), 3)
    xs = xyz[:, 0]
    left, right = xyz[xs < 0], xyz[xs > 0]
    assert len(left) >= 8 and len(right) >= 8
    assert left[:, 0].mean() == pytest.approx(-1.0, abs=0.2)
    assert right[:, 0].mean() == pytest.approx(1.0, abs=0.2)


def test_second_camera_extends_cloud_forward():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = _corridor_depth(K)
    rgb = _gray_rgb(K.height, K.width)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[2, 3] = 2.0
    fused = fuse_frames(
        [(depth, rgb, T1), (depth, rgb, T2)],
        K, stride=4, min_depth=0.3, max_depth=15.0, voxel=0.10)
    first, _ = unproject_frame(depth, rgb, K, T1, stride=4)
    assert fused.xyz[:, 2].max() > first[:, 2].max() + 1.0


def test_voxel_downsample_reduces_count_and_keeps_cell():
    xyz = np.array([[0.01, 0.0, 1.0], [0.02, 0.0, 1.0], [1.0, 0.0, 1.0]],
                   dtype=np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    out_xyz, out_rgb = voxel_downsample(xyz, rgb, 0.10)
    assert len(out_xyz) == 2
    assert out_xyz.shape == out_rgb.shape
