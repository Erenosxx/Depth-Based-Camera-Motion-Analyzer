import json

import pytest

from dcma.cli import build_parser, run


def test_parser_requires_scene():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "x.mp4", "--out", "o"])


def test_parser_rejects_unknown_scene():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "x.mp4", "--out", "o", "--scene", "space"])


def test_parser_defaults():
    args = build_parser().parse_args(
        ["--video", "x.mp4", "--out", "o", "--scene", "indoor"])
    assert args.size == "large"
    assert args.fov_long == 70.0
    assert args.interval == 0.15
    assert args.no_map is False


def test_parser_no_map_flag():
    args = build_parser().parse_args(
        ["--video", "x.mp4", "--out", "o", "--scene", "indoor", "--no-map"])
    assert args.no_map is True


@pytest.mark.needs_video
@pytest.mark.needs_gpu
def test_end_to_end_writes_trajectory(sample_video, ffmpeg_available, cuda_device, tmp_path):
    args = build_parser().parse_args([
        "--video", sample_video, "--out", str(tmp_path),
        "--scene", "indoor", "--size", "base",
        "--max-frames", "24", "--max-edge", "512",
    ])
    run(args)

    path = tmp_path / "trajectory.json"
    assert path.is_file()
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["step_count"] > 0
    assert "odometer" in d
    assert all(k in d["odometer"] for k in ("forward", "right", "up"))
    assert "poses" in d
    assert (tmp_path / "occupancy.npz").is_file()
    assert (tmp_path / "occupancy.png").is_file()
    assert (tmp_path / "map.ply").is_file()
    assert (tmp_path / "map_preview.png").is_file()
