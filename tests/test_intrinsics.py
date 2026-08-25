import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics


def base():
    return Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)


def test_matrix_layout():
    K = base().matrix
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, [[1000.0, 0.0, 640.0],
                                   [0.0, 1000.0, 360.0],
                                   [0.0, 0.0, 1.0]])


def test_scaled_half_halves_every_term():
    s = base().scaled(0.5)
    assert (s.fx, s.fy, s.cx, s.cy) == (500.0, 500.0, 320.0, 180.0)
    assert (s.width, s.height) == (640, 360)


def test_scaled_to_max_edge_computes_factor():
    s = base().scaled_to_max_edge(640)
    assert (s.width, s.height) == (640, 360)
    assert s.fx == pytest.approx(500.0)


def test_cropped_shifts_principal_point_only():
    c = base().cropped(x0=100, y0=50, width=1000, height=600)
    assert (c.fx, c.fy) == (1000.0, 1000.0)
    assert (c.cx, c.cy) == (540.0, 310.0)
    assert (c.width, c.height) == (1000, 600)


def test_scale_then_crop_matches_manual_composition():
    k = base().scaled(0.5).cropped(x0=20, y0=10, width=600, height=340)
    assert (k.fx, k.cx, k.cy) == (500.0, 300.0, 170.0)


def test_from_fov_horizontal():
    # 90 derece yatay FOV -> fx = (w/2) / tan(45deg) = w/2
    k = Intrinsics.from_fov(width=1000, height=1000, fov_x_deg=90.0)
    assert k.fx == pytest.approx(500.0)
    assert k.cx == pytest.approx(500.0)
    assert k.cy == pytest.approx(500.0)


def test_rotated_90_swaps_axes_and_maps_principal_point():
    # (u, v) -> (H-1-v, u) saat yonunde 90 derece
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    r = k.rotated(90)
    assert (r.fx, r.fy) == (200.0, 100.0)
    assert (r.cx, r.cy) == (9.0, 10.0)      # (30-1)-20 = 9,  cx = 10
    assert (r.width, r.height) == (30, 40)


def test_rotated_270_is_inverse_of_90():
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    assert k.rotated(90).rotated(270) == k


def test_rotated_180_mirrors_principal_point():
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    r = k.rotated(180)
    assert (r.fx, r.fy) == (100.0, 200.0)
    assert (r.cx, r.cy) == (29.0, 9.0)      # (40-1)-10 = 29, (30-1)-20 = 9
    assert (r.width, r.height) == (40, 30)


def test_rotated_zero_is_identity():
    k = base()
    assert k.rotated(0) == k


def test_rotated_rejects_non_multiples_of_90():
    with pytest.raises(ValueError):
        base().rotated(45)


def test_json_roundtrip():
    k = base()
    assert Intrinsics.from_dict(k.to_dict()) == k


def test_rejects_nonpositive_focal():
    with pytest.raises(ValueError):
        Intrinsics(fx=0.0, fy=1000.0, cx=1.0, cy=1.0, width=10, height=10)


def test_crop_outside_bounds_rejected():
    with pytest.raises(ValueError):
        base().cropped(x0=1000, y0=0, width=1000, height=600)
