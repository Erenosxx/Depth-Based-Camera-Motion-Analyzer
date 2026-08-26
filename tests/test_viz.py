import json

import cv2
import numpy as np

from dcma.viz.annotate import draw_overlay, write_annotated_video
from dcma.viz.plot import write_trajectory_plot


def _payload():
    return {
        "step_count": 2,
        "odometer": {"forward": 0.3, "right": 0.1, "up": 0.0},
        "net_displacement": {
            "forward": 0.3, "right": 0.1, "up": 0.0, "magnitude": 0.316,
        },
        "path_length": 0.3,
        "positions": [[0.0, 0.0, 0.0], [0.05, 0.0, 0.15], [0.1, 0.0, 0.3]],
        "steps": [
            {
                "frame_from": 0, "frame_to": 1,
                "forward": 0.15, "right": 0.05, "up": 0.0,
                "distance": 0.158, "inliers": 80, "reproj_err": 0.5,
            },
            {
                "frame_from": 1, "frame_to": 2,
                "forward": 0.15, "right": 0.05, "up": 0.0,
                "distance": 0.158, "inliers": 90, "reproj_err": 0.4,
            },
        ],
        "skipped": [],
        "manifest": {"source_fps": 30.0, "frame_count": 3},
    }


def test_draw_overlay_keeps_shape_and_changes_pixels():
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    p1 = np.array([[20.0, 30.0], [80.0, 50.0]], dtype=np.float32)
    p2 = np.array([[40.0, 36.0], [95.0, 58.0]], dtype=np.float32)
    out = draw_overlay(
        frame,
        p1=p1,
        p2=p2,
        alpha=0.5,
        odo={"forward": 1.2, "right": -0.3, "up": 0.1},
        step={"forward": 0.08, "right": 0.0, "up": 0.0, "inliers": 40, "reproj_err": 0.7},
        path_xy=np.array([[0.0, 0.0], [0.1, 0.4]]),
        frame_idx=12,
    )
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_track_interpolation_moves_the_dot():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    p1 = np.array([[80.0, 120.0]], dtype=np.float32)
    p2 = np.array([[240.0, 120.0]], dtype=np.float32)
    kwargs = dict(
        odo={"forward": 0, "right": 0, "up": 0},
        step={"forward": 0.08, "right": 0, "up": 0},
        path_xy=np.array([[0.0, 0.0]]),
    )
    a = draw_overlay(frame, p1=p1, p2=p2, alpha=0.0, frame_idx=0, **kwargs)
    b = draw_overlay(frame, p1=p1, p2=p2, alpha=1.0, frame_idx=1, **kwargs)
    assert a[120, 80].sum() > 0
    assert a[120, 240].sum() == 0
    assert b[120, 240].sum() > 0


def test_write_trajectory_plot_creates_png(tmp_path):
    path = tmp_path / "plot.png"
    write_trajectory_plot(_payload(), path)
    assert path.is_file()
    assert path.stat().st_size > 1000
    img = cv2.imread(str(path))
    assert img is not None
    assert img.ndim == 3


def test_write_annotated_video_from_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        img = np.full((96, 128, 3), 30 + i * 10, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i + 1:06d}.png"), img)

    payload = _payload()
    out = tmp_path / "annotated.mp4"
    written = write_annotated_video(payload, frames_dir, out)
    assert written.is_file()
    assert written.stat().st_size > 0

    cap = cv2.VideoCapture(str(written))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame.shape[0] == 96
    assert frame.shape[1] == 128
