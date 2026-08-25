import pytest

from dcma.video.normalize import probe_video, normalize_video


@pytest.mark.needs_video
def test_probe_reports_geometry_and_rotation(sample_video, ffmpeg_available):
    info = probe_video(sample_video)
    assert info.width > 0 and info.height > 0
    assert info.fps > 0
    assert info.codec
    assert info.rotation_deg in (0, 90, 180, 270)


@pytest.mark.needs_video
def test_normalize_writes_png_frames_and_manifest(sample_video, ffmpeg_available, tmp_path):
    m = normalize_video(sample_video, tmp_path, max_frames=8)
    frames = sorted((tmp_path / "frames").glob("*.png"))
    assert len(frames) == 8
    assert m.frame_count == 8
    assert m.frame_format == "png"
    assert m.rotation_applied is True
    assert (tmp_path / "manifest.json").is_file()


@pytest.mark.needs_video
def test_rotation_changes_frame_geometry_when_not_square(sample_video, ffmpeg_available, tmp_path):
    """Dönme uygulandığında kare boyutu kaynağa göre takla atmış olmalı (kare olmayan video)."""
    info = probe_video(sample_video)
    m = normalize_video(sample_video, tmp_path, max_frames=2)
    if info.rotation_deg in (90, 270) and info.width != info.height:
        assert (m.frame_width, m.frame_height) == (info.height, info.width)
    else:
        assert (m.frame_width, m.frame_height) == (info.width, info.height)


@pytest.mark.needs_video
def test_max_edge_downscales_and_updates_intrinsics(sample_video, ffmpeg_available, tmp_path):
    from dcma.calib.intrinsics import Intrinsics
    info = probe_video(sample_video)
    k = Intrinsics.from_fov(info.width, info.height, 70.0)
    m = normalize_video(sample_video, tmp_path, max_frames=2, max_edge=320, intrinsics=k)
    assert max(m.frame_width, m.frame_height) == 320
    assert m.intrinsics is not None
    assert m.intrinsics.width == m.frame_width
    assert m.intrinsics.fx < k.fx
