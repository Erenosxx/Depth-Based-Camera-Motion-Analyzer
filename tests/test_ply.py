from __future__ import annotations

import numpy as np

from dcma.map.ply import read_ply_xyz_rgb, write_ply_xyz_rgb


def test_ply_roundtrip(tmp_path):
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, -0.5, 2.0]], dtype=np.float64)
    rgb = np.array([[10, 20, 30], [255, 0, 1]], dtype=np.uint8)
    path = tmp_path / "map.ply"
    write_ply_xyz_rgb(path, xyz, rgb)
    text = path.read_text(encoding="ascii")
    assert "element vertex 2" in text
    assert "property uchar red" in text
    xyz2, rgb2 = read_ply_xyz_rgb(path)
    np.testing.assert_allclose(xyz2, xyz)
    np.testing.assert_array_equal(rgb2, rgb)
