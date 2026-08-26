import cv2
import numpy as np
import pytest

from dcma.viz.annotate import (
    EDGE_PAD, _direction, draw_overlay, minimap_view, write_annotated_video,
)
from dcma.viz.plot import write_trajectory_plot
from dcma.viz.tracks import PointTracker, color_for_id
from dcma.map.occupancy import OccupancyGrid


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


def _textured(size=160, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 30, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(10, size - 40, 2)
        w, h = rng.integers(8, 28, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      int(rng.integers(140, 255)), -1)
    return img


def _norm(pt, center, span):
    return (pt - center) / (2 * span) + 0.5


def test_minimap_centers_green_and_red_not_origin():
    path = np.array([[0.0, 0.0], [0.0, 4.0]])
    center, span = minimap_view(path)
    np.testing.assert_allclose(center, [0.0, 2.0])
    g = _norm(path[0], center, span)
    r = _norm(path[-1], center, span)
    assert g[1] == pytest.approx(EDGE_PAD)
    assert r[1] == pytest.approx(1.0 - EDGE_PAD)
    assert EDGE_PAD < 0.15


def test_minimap_straight_run_keeps_same_on_screen_size():
    c1, s1 = minimap_view(np.array([[0.0, 0.0], [0.0, 1.0]]))
    c4, s4 = minimap_view(np.array([[0.0, 0.0], [0.0, 4.0]]))
    assert _norm(np.array([0.0, 0.0]), c1, s1)[1] == pytest.approx(
        _norm(np.array([0.0, 0.0]), c4, s4)[1])
    assert _norm(np.array([0.0, 1.0]), c1, s1)[1] == pytest.approx(1.0 - EDGE_PAD)
    assert _norm(np.array([0.0, 4.0]), c4, s4)[1] == pytest.approx(1.0 - EDGE_PAD)


def test_minimap_freezes_when_returning_toward_green():
    outbound = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    returning = np.vstack([outbound, [[0.0, 2.5], [0.0, 1.0]]])
    c_out, s_out = minimap_view(outbound)
    c_back, s_back = minimap_view(returning)
    np.testing.assert_allclose(c_out, c_back)
    assert s_back == pytest.approx(s_out)
    _, s_short = minimap_view(np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert s_back > s_short


def test_minimap_resumes_when_farther_than_record():
    path = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 1.0], [0.0, 6.0]])
    c, s = minimap_view(path)
    c_far, s_far = minimap_view(np.array([[0.0, 0.0], [0.0, 6.0]]))
    np.testing.assert_allclose(c, c_far)
    assert s == pytest.approx(s_far)


def test_minimap_keeps_sideways_leg_on_screen():
    """Yeşil–en uzak kirişi L kolunu ekran dışına atıyordu."""
    path = np.array([[0.0, 0.0], [5.0, 5.0], [6.0, 0.0]])
    center, span = minimap_view(path)
    for pt in path:
        n = _norm(pt, center, span)
        assert EDGE_PAD - 1e-6 <= n[0] <= 1.0 - EDGE_PAD + 1e-6
        assert EDGE_PAD - 1e-6 <= n[1] <= 1.0 - EDGE_PAD + 1e-6


def test_minimap_occupancy_keeps_current_inside():
    path = np.array([[0.0, 0.0], [0.0, 2.0]])
    occ = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 8.0]])
    center, span = minimap_view(path, extras=occ)
    n_red = _norm(path[-1], center, span)
    assert 0.05 < n_red[0] < 0.95
    assert 0.05 < n_red[1] < 0.95
    n_wall = _norm(np.array([0.0, 8.0]), center, span)
    assert EDGE_PAD - 1e-6 <= n_wall[1] <= 1.0 - EDGE_PAD + 1e-6


def test_minimap_heading_lookahead_pulls_frontier_inward():
    """Bakış yönünde pay: kırmızı uçta ok kesilmesin."""
    path = np.array([[0.0, 0.0], [0.0, 4.0]])
    c, s = minimap_view(path)
    n0 = _norm(path[-1], c, s)
    assert n0[1] == pytest.approx(1.0 - EDGE_PAD)
    c2, s2 = minimap_view(path, heading_xy=np.array([0.0, 1.0]))
    n1 = _norm(path[-1], c2, s2)
    assert n1[1] < n0[1] - 0.03
    assert _norm(path[0], c2, s2)[1] >= EDGE_PAD - 1e-6


def test_ids_get_distinct_colors():
    colors = {color_for_id(i) for i in range(20)}
    assert len(colors) >= 15
    assert (0, 255, 255) not in colors or len(colors) > 1


def test_tracker_trail_follows_l_shape_not_diagonal():
    gray = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(gray, (20, 20), (40, 40), 255, -1)
    tr = PointTracker(max_points=4, trail_len=20, min_distance=8)
    tr.seed_at(gray, np.array([[30.0, 30.0]], dtype=np.float32))

    x, y = 30.0, 30.0
    for _ in range(8):
        x += 4
        gray = np.zeros((120, 160), dtype=np.uint8)
        cv2.rectangle(gray, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), 255, -1)
        tr.update(gray)
    for _ in range(8):
        y += 4
        gray = np.zeros((120, 160), dtype=np.uint8)
        cv2.rectangle(gray, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), 255, -1)
        tr.update(gray)

    assert tr.trails
    path = np.array(tr.trails[0])
    assert len(path) >= 8
    # L: once x artar y sabit, sonra y artar — kosegen (x ve y birlikte) olmamali
    mid = len(path) // 2
    dx_first = path[mid, 0] - path[0, 0]
    dy_first = abs(path[mid, 1] - path[0, 1])
    dy_last = path[-1, 1] - path[mid, 1]
    assert dx_first > 8
    assert dy_first < dx_first
    assert dy_last > 8


def test_state_fills_yaw_from_poses_when_step_omits_it():
    """Eski trajectory.json yaw_deg yazmaz; pozlardan SAĞA DÖN üretilmeli."""
    from dcma.vo.trajectory import Trajectory
    from dcma.viz.annotate import _state_at_frame

    tr = Trajectory()
    R = cv2.Rodrigues(np.array([0.0, np.deg2rad(-12.0), 0.0]))[0]
    tr.add_step(R, np.zeros(3), frame_from=0, frame_to=5)
    payload = tr.to_dict()
    payload["steps"][0].pop("yaw_deg")
    _, step, _, heading = _state_at_frame(payload, 3)
    assert _direction(step) == "SAĞA DÖN"
    assert step["yaw_deg"] == pytest.approx(-12.0, abs=1.0)
    assert heading is not None
    assert float(np.linalg.norm(heading)) > 0.1


def test_direction_reports_right_turn_not_forward():
    step = {
        "forward": 0.20, "right": 0.0, "up": 0.0,
        "yaw_deg": -12.0,
    }
    assert _direction(step) == "SAĞA DÖN"


def test_draw_overlay_keeps_shape_and_paints_hud():
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    out = draw_overlay(
        frame,
        odo={"forward": 1.2, "right": -0.3, "up": 0.1},
        step={"forward": 0.08, "right": 0.0, "up": 0.0, "inliers": 40, "reproj_err": 0.7},
        path_xy=np.array([[0.0, 0.0], [0.1, 0.4]]),
        frame_idx=12,
    )
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_minimap_paints_occupancy_cells():
    frame = np.full((200, 200, 3), 40, dtype=np.uint8)
    kwargs = dict(
        odo={"forward": 0.0, "right": 0.0, "up": 0.0},
        step={"forward": 0.05, "right": 0.0, "up": 0.0},
        path_xy=np.array([[0.0, 0.0], [0.0, 2.0]]),
        frame_idx=0,
    )
    without = draw_overlay(frame, **kwargs)
    with_occ = draw_overlay(
        frame, **kwargs, occ_xy=np.array([[-0.8, 1.0], [0.8, 1.0]]),
    )
    assert with_occ.shape == without.shape
    assert not np.array_equal(with_occ, without)


def test_write_trajectory_plot_creates_png(tmp_path):
    path = tmp_path / "plot.png"
    write_trajectory_plot(_payload(), path)
    assert path.is_file()
    assert path.stat().st_size > 1000
    img = cv2.imread(str(path))
    assert img is not None
    assert img.ndim == 3


def test_write_trajectory_plot_accepts_occupancy(tmp_path):
    from dcma.calib.intrinsics import Intrinsics
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = np.full((120, 160), 4.0, dtype=np.float32)
    occ = OccupancyGrid(resolution=0.10, stride=4, y_lo=-2.0, y_hi=2.0)
    occ.splat(depth, K, np.eye(4), frame_idx=0)
    path = tmp_path / "plot.png"
    write_trajectory_plot(_payload(), path, occupancy=occ)
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_write_annotated_video_fps_matches_manifest(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    base = cv2.cvtColor(_textured(), cv2.COLOR_GRAY2BGR)
    n = 15
    fps = 30.0
    for i in range(n):
        shift = np.roll(base, i * 2, axis=1)
        cv2.imwrite(str(frames_dir / f"frame_{i + 1:06d}.png"), shift)

    payload = _payload()
    payload["manifest"] = {"source_fps": fps, "frame_count": n}
    payload["steps"][0]["frame_from"] = 0
    payload["steps"][0]["frame_to"] = 5
    payload["steps"][1]["frame_from"] = 5
    payload["steps"][1]["frame_to"] = 10
    out = tmp_path / "annotated.mp4"
    written = write_annotated_video(payload, frames_dir, out)
    assert written.is_file()
    assert written.stat().st_size > 0

    cap = cv2.VideoCapture(str(written))
    assert cap.isOpened()
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert got_fps == pytest.approx(fps, abs=0.5)
    assert count == n
