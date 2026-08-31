"""Sentetik koridor: sol/sağ duvarlar dünya x = ±1 m, derinlik kamera z."""

from __future__ import annotations

import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.map.occupancy import OccupancyGrid


def _corridor_depth(K: Intrinsics, wall_x: float = 1.0) -> np.ndarray:
    """Her sütun ilgili duvara (x=±wall_x) çarpsın; orta satırlar duvar diliminde."""
    depth = np.full((K.height, K.width), np.nan, dtype=np.float32)
    us = np.arange(K.width, dtype=np.float64)
    left = us < K.cx
    right = us > K.cx
    z_left = (-wall_x) * K.fx / (us[left] - K.cx)
    z_right = wall_x * K.fx / (us[right] - K.cx)
    depth[:, left] = z_left[np.newaxis, :]
    depth[:, right] = z_right[np.newaxis, :]
    return depth


def test_identity_pose_marks_two_parallel_walls():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4)
    grid.splat(_corridor_depth(K), K, np.eye(4), frame_idx=0)
    cells = grid.cells_upto(0)
    assert len(cells) >= 8
    xs = cells[:, 0]
    left = cells[xs < 0]
    right = cells[xs > 0]
    assert len(left) >= 3 and len(right) >= 3
    assert left[:, 0].mean() == pytest.approx(-1.0, abs=0.2)
    assert right[:, 0].mean() == pytest.approx(1.0, abs=0.2)


def test_second_frame_extends_grid_but_cells_upto_zero_excludes_it():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = _corridor_depth(K)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4)
    grid.splat(depth, K, np.eye(4), frame_idx=0)
    first = grid.cells_upto(0)
    T = np.eye(4)
    T[2, 3] = 2.0
    grid.splat(depth, K, T, frame_idx=10)
    later = grid.cells_upto(10)
    assert later[:, 1].max() > first[:, 1].max() + 1.0
    again_first = grid.cells_upto(0)
    np.testing.assert_allclose(again_first, first)


def test_npz_roundtrip_preserves_cells_upto(tmp_path):
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4)
    grid.splat(_corridor_depth(K), K, np.eye(4), frame_idx=3)
    path = tmp_path / "occupancy.npz"
    grid.to_npz(path)
    loaded = OccupancyGrid.from_npz(path)
    np.testing.assert_allclose(loaded.cells_upto(3), grid.cells_upto(3))
    assert loaded.resolution == pytest.approx(0.10)


def test_write_png_creates_image(tmp_path):
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4)
    grid.splat(_corridor_depth(K), K, np.eye(4), frame_idx=0)
    png = tmp_path / "occupancy.png"
    grid.write_png(png)
    assert png.is_file()
    assert png.stat().st_size > 500


def test_splat_skips_points_too_close_to_camera():
    """Kameranın hemen önündeki yakın derinlik 'duvar' olmasın (yana kayma artefaktı)."""
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = np.full((K.height, K.width), 0.35, dtype=np.float32)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4,
                         min_depth=0.3, near_m=0.8)
    grid.splat(depth, K, np.eye(4), frame_idx=0)
    assert len(grid.cells_upto(0)) == 0


def test_splat_keeps_far_wall_in_front():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = np.full((K.height, K.width), 3.0, dtype=np.float32)
    grid = OccupancyGrid(resolution=0.10, stride=4, y_lo=-0.4, y_hi=0.4,
                         min_depth=0.3, near_m=0.8)
    grid.splat(depth, K, np.eye(4), frame_idx=0)
    cells = grid.cells_upto(0)
    assert len(cells) >= 4
    assert cells[:, 1].mean() == pytest.approx(3.0, abs=0.3)
    # Karşı duvarın görüntü ortası atılır; hit'ler yanda kalır.
    assert np.all(np.abs(cells[:, 0]) > 0.15)
