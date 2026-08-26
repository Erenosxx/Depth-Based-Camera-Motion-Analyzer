import cv2
import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.vo.pose import backproject, estimate_pose


def _scene(R, t, n=400, seed=0):
    """Bilinen (R, t) ile sentetik sahne: 3B noktalar ve iki karedeki izdüşümleri."""
    rng = np.random.default_rng(seed)
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0,
                   width=1000, height=1000)
    X1 = np.stack([rng.uniform(-2.0, 2.0, n),
                   rng.uniform(-2.0, 2.0, n),
                   rng.uniform(2.0, 6.0, n)], axis=1)
    X2 = (R @ X1.T).T + t

    def proj(X):
        return np.stack([K.fx * X[:, 0] / X[:, 2] + K.cx,
                         K.fy * X[:, 1] / X[:, 2] + K.cy], axis=1).astype(np.float32)

    return K, X1, proj(X1), proj(X2)


def test_pure_forward_translation_recovers_exact_metres():
    d = 0.25
    R = np.eye(3)
    t = np.array([0.0, 0.0, -d])       # kamera +z'de d ilerlerse noktalar -d kayar
    K, X1, _, p2 = _scene(R, t)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.inliers > 350
    np.testing.assert_allclose(res.R, R, atol=1e-6)
    np.testing.assert_allclose(res.t, t, atol=1e-6)
    np.testing.assert_allclose(res.C, [0.0, 0.0, d], atol=1e-6)
    assert res.forward == pytest.approx(d, abs=1e-6)
    assert res.right == pytest.approx(0.0, abs=1e-6)
    assert res.up == pytest.approx(0.0, abs=1e-6)


def test_pure_right_translation_reports_positive_right():
    d = 0.4
    R = np.eye(3)
    t = np.array([-d, 0.0, 0.0])       # kamera +x'te (saga) d giderse noktalar -d kayar
    K, X1, _, p2 = _scene(R, t, seed=3)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.right == pytest.approx(d, abs=1e-6)
    assert res.forward == pytest.approx(0.0, abs=1e-6)


def test_pure_up_translation_reports_positive_up():
    d = 0.3
    R = np.eye(3)
    t = np.array([0.0, d, 0.0])        # y ASAGI oldugu icin yukari hareket +y kaydirir
    K, X1, _, p2 = _scene(R, t, seed=4)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.up == pytest.approx(d, abs=1e-6)


def test_rotation_and_translation_recovered_together():
    R = cv2.Rodrigues(np.array([0.01, np.deg2rad(4.0), -0.02]))[0]
    t = np.array([0.05, -0.02, -0.30])
    K, X1, _, p2 = _scene(R, t, seed=7)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    np.testing.assert_allclose(res.R, R, atol=1e-6)
    np.testing.assert_allclose(res.t, t, atol=1e-6)
    assert res.reproj_err < 1e-3


def test_backproject_inverts_projection():
    K = Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)
    X = np.array([[0.5, -0.25, 3.0], [-1.0, 0.75, 5.0]])
    uv = np.stack([K.fx * X[:, 0] / X[:, 2] + K.cx,
                   K.fy * X[:, 1] / X[:, 2] + K.cy], axis=1).astype(np.float32)
    depth = np.zeros((480, 640), dtype=np.float32)
    for (u, v), z in zip(np.round(uv).astype(int), X[:, 2]):
        depth[v, u] = z

    P3d, z = backproject(depth, K, uv)
    np.testing.assert_allclose(z, X[:, 2], atol=1e-4)
    np.testing.assert_allclose(P3d, X, atol=1e-2)


def test_yaw_dominant_relative_pose_drops_translation():
    """Yerinde dönüş (~15°/adım) yürüyüş ötelemesi sanılmasın."""
    from dcma.vo.pose import maybe_inplace_yaw
    R = cv2.Rodrigues(np.array([0.0, np.deg2rad(-15.0), 0.0]))[0]
    t = np.array([0.05, -0.02, -0.30])
    Rg, tg = maybe_inplace_yaw(R, t)
    np.testing.assert_allclose(Rg, R)
    np.testing.assert_allclose(tg, 0.0, atol=1e-12)


def test_small_yaw_keeps_translation():
    from dcma.vo.pose import maybe_inplace_yaw
    R = cv2.Rodrigues(np.array([0.0, np.deg2rad(-3.0), 0.0]))[0]
    t = np.array([0.0, 0.0, -0.25])
    _, tg = maybe_inplace_yaw(R, t)
    np.testing.assert_allclose(tg, t)


def test_too_few_points_returns_none():
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    res = estimate_pose(np.zeros((3, 3)), np.zeros((3, 2), dtype=np.float32), K)
    assert res is None
