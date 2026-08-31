from __future__ import annotations

import numpy as np

from dcma.map.preview import write_map_preview


def test_write_map_preview_creates_png(tmp_path):
    xyz = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.2, 2.0],
        [0.5, -0.1, 1.5],
    ], dtype=np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    path = tmp_path / "map_preview.png"
    out = write_map_preview(xyz, rgb, path)
    assert out.is_file()
    assert out.stat().st_size > 500
