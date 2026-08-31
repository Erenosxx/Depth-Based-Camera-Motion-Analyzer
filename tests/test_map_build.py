from __future__ import annotations

import json

import numpy as np
from PIL import Image

from dcma.calib.intrinsics import Intrinsics
from dcma.map.build import build_map
from dcma.map.ply import read_ply_xyz_rgb
from dcma.vo.trajectory import Trajectory


def test_build_map_from_run_dir(tmp_path):
    K = Intrinsics(fx=100.0, fy=100.0, cx=4.5, cy=3.5, width=10, height=8)
    frames = tmp_path / "frames"
    cache = tmp_path / "depth_cache"
    frames.mkdir()
    cache.mkdir()
    rgb = np.full((8, 10, 3), 40, dtype=np.uint8)
    Image.fromarray(rgb).save(frames / "000000.png")
    Image.fromarray(rgb).save(frames / "000003.png")
    depth = np.full((8, 10), 2.0, dtype=np.float16)
    np.save(cache / "000000.npy", depth)
    np.save(cache / "000003.npy", depth)

    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, 0.5]), frame_from=0, frame_to=3)
    payload = tr.to_dict()
    payload["manifest"] = {
        "source_path": "x.mp4",
        "source_width": 10,
        "source_height": 8,
        "source_fps": 30.0,
        "source_codec": "h264",
        "rotation_tag_deg": 0,
        "rotation_applied": False,
        "frames_dir": str(frames),
        "frame_count": 2,
        "frame_width": 10,
        "frame_height": 8,
        "frame_format": "png",
        "transforms": [],
        "intrinsics": K.to_dict(),
    }
    (tmp_path / "trajectory.json").write_text(
        json.dumps(payload), encoding="utf-8")

    out = build_map(tmp_path, voxel=0.05, stride=2)
    assert out["ply"].is_file()
    xyz, _rgb = read_ply_xyz_rgb(out["ply"])
    assert len(xyz) > 0
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))
    assert meta["n_points"] == len(xyz)
    assert meta["n_frames"] == 2
    assert out["preview"].is_file()
    assert out["preview"].stat().st_size > 500


def test_build_map_resolves_ffmpeg_frame_percent_names(tmp_path):
    """normalize.py kareleri frame_%06d.png (1-tabanlı) yazar; cache anahtarı liste indeksi."""
    K = Intrinsics(fx=100.0, fy=100.0, cx=4.5, cy=3.5, width=10, height=8)
    frames = tmp_path / "frames"
    cache = tmp_path / "depth_cache"
    frames.mkdir()
    cache.mkdir()
    rgb = np.full((8, 10, 3), 40, dtype=np.uint8)
    Image.fromarray(rgb).save(frames / "frame_000001.png")
    Image.fromarray(rgb).save(frames / "frame_000004.png")
    depth = np.full((8, 10), 2.0, dtype=np.float16)
    np.save(cache / "000000.npy", depth)
    np.save(cache / "000001.npy", depth)

    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, 0.5]), frame_from=0, frame_to=1)
    payload = tr.to_dict()
    payload["manifest"] = {
        "source_path": "x.mp4",
        "source_width": 10,
        "source_height": 8,
        "source_fps": 30.0,
        "source_codec": "h264",
        "rotation_tag_deg": 0,
        "rotation_applied": False,
        "frames_dir": str(frames),
        "frame_count": 2,
        "frame_width": 10,
        "frame_height": 8,
        "frame_format": "png",
        "transforms": [],
        "intrinsics": K.to_dict(),
    }
    (tmp_path / "trajectory.json").write_text(
        json.dumps(payload), encoding="utf-8")

    out = build_map(tmp_path, voxel=0.05, stride=2)
    assert out["n_points"] > 0
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))
    assert meta["n_frames"] == 2
    assert meta["skipped"] == []
