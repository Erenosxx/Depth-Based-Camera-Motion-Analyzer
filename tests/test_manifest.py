import json

import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.video.manifest import VideoManifest


def sample(tmp_path):
    return VideoManifest(
        source_path="/videos/in.mp4",
        source_width=1440,
        source_height=1440,
        source_fps=30.0,
        source_codec="hevc",
        rotation_tag_deg=90,
        rotation_applied=True,
        frames_dir=str(tmp_path / "frames"),
        frame_count=207,
        frame_width=1440,
        frame_height=1440,
        frame_format="png",
        transforms=[{"kind": "autorotate", "degrees": 90}],
        intrinsics=Intrinsics(fx=1100.0, fy=1100.0, cx=720.0, cy=720.0,
                              width=1440, height=1440),
    )


def test_json_roundtrip_preserves_every_field(tmp_path):
    m = sample(tmp_path)
    path = tmp_path / "manifest.json"
    m.save(path)
    assert VideoManifest.load(path) == m


def test_saved_json_is_human_readable(tmp_path):
    m = sample(tmp_path)
    path = tmp_path / "manifest.json"
    m.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["rotation_applied"] is True
    assert d["intrinsics"]["fx"] == 1100.0


def test_intrinsics_may_be_absent(tmp_path):
    m = sample(tmp_path)
    m2 = VideoManifest(**{**m.to_dict(), "intrinsics": None})
    path = tmp_path / "m.json"
    m2.save(path)
    assert VideoManifest.load(path).intrinsics is None


def test_frame_seconds_uses_fps(tmp_path):
    m = sample(tmp_path)
    assert m.frame_seconds(0) == pytest.approx(0.0)
    assert m.frame_seconds(30) == pytest.approx(1.0)


def test_stride_for_interval_rounds_to_at_least_one(tmp_path):
    m = sample(tmp_path)
    assert m.stride_for_interval(0.1) == 3      # 30 fps * 0.1 s
    assert m.stride_for_interval(0.001) == 1    # asla 0 olmaz
