import numpy as np
import pytest

from dcma.video.normalize import normalize_video, probe_video
from dcma.video.orientation import cv2_applies_rotation, read_frame_upright


@pytest.mark.needs_video
def test_report_whether_cv2_autorotates(sample_video, ffmpeg_available, capsys):
    """Bu test bir davranışı sabitlemez, ÖLÇER ve kayda geçirir."""
    info = probe_video(sample_video)
    applies = cv2_applies_rotation(sample_video)
    with capsys.disabled():
        print(f"\n  rotation_tag={info.rotation_deg}  cv2_autorotate={applies}")
    assert isinstance(applies, bool)


@pytest.mark.needs_video
def test_read_frame_upright_matches_ffmpeg_frame(sample_video, ffmpeg_available, tmp_path):
    """read_frame_upright, ffmpeg'in autorotate çıktısıyla birebir aynı olmalı."""
    import cv2

    m = normalize_video(sample_video, tmp_path, max_frames=5)
    ff = cv2.imread(str(sorted((tmp_path / "frames").glob("*.png"))[3]))
    assert ff is not None

    got = read_frame_upright(sample_video, index=3)
    assert got is not None
    assert got.shape == ff.shape, f"boyut uyuşmazlığı: {got.shape} != {ff.shape}"
    assert np.abs(got.astype(int) - ff.astype(int)).mean() < 1.0
