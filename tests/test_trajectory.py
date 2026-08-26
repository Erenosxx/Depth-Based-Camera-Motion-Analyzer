import cv2
import numpy as np
import pytest

from dcma.vo.trajectory import Trajectory


def test_starts_at_origin_with_identity_pose():
    tr = Trajectory()
    np.testing.assert_allclose(tr.positions[-1], [0.0, 0.0, 0.0])
    assert tr.step_count == 0


def test_two_forward_steps_accumulate():
    tr = Trajectory()
    for _ in range(2):
        tr.add_step(np.eye(3), np.array([0.0, 0.0, -0.1]))

    np.testing.assert_allclose(tr.positions[-1], [0.0, 0.0, 0.2], atol=1e-9)
    odo = tr.odometer
    assert odo["forward"] == pytest.approx(0.2)
    assert odo["right"] == pytest.approx(0.0)
    assert odo["up"] == pytest.approx(0.0)
    np.testing.assert_allclose(tr.net_displacement, [0.0, 0.0, 0.2], atol=1e-9)


def test_yaw_only_step_records_right_turn():
    """Yerinde sağa bakış (ofis gezisinde yaw_deg negatif) step'e yazılır."""
    tr = Trajectory()
    R = cv2.Rodrigues(np.array([0.0, np.deg2rad(-90.0), 0.0]))[0]
    tr.add_step(R, np.zeros(3))
    yaw = tr.to_dict()["steps"][0]["yaw_deg"]
    assert yaw == pytest.approx(-90.0, abs=1.0)


def test_accumulation_equals_manual_matrix_product():
    """Birikim tam olarak T_{k+1} = T_k @ inv(dT_k) olmalı."""
    rng = np.random.default_rng(11)
    steps = []
    for _ in range(4):
        rvec = rng.normal(scale=0.05, size=3)
        R = cv2.Rodrigues(rvec)[0]
        t = rng.normal(scale=0.2, size=3)
        steps.append((R, t))

    tr = Trajectory()
    for R, t in steps:
        tr.add_step(R, t)

    expected = np.eye(4)
    for R, t in steps:
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        expected = expected @ np.linalg.inv(dT)

    np.testing.assert_allclose(tr.poses[-1], expected, atol=1e-10)


def test_loop_returns_near_origin_but_odometer_grows():
    """Ileri git, 180 don, ayni kadar ileri git -> baslangica yakin don."""
    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -1.0]))
    R180 = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0]
    tr.add_step(R180, np.array([0.0, 0.0, 0.0]))
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -1.0]))

    assert np.linalg.norm(tr.net_displacement) < 1e-6
    assert tr.odometer["forward"] == pytest.approx(2.0, abs=1e-9)
    assert tr.path_length == pytest.approx(2.0, abs=1e-9)


def test_export_dict_has_per_step_records():
    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -0.1]), inliers=120, reproj_err=0.7)
    d = tr.to_dict()
    assert d["step_count"] == 1
    assert d["steps"][0]["inliers"] == 120
    assert d["steps"][0]["forward"] == pytest.approx(0.1)
    assert "odometer" in d and "net_displacement" in d


def test_export_dict_includes_world_poses():
    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -0.1]))
    d = tr.to_dict()
    poses = np.asarray(d["poses"], dtype=np.float64)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4))
    np.testing.assert_allclose(poses[-1], tr.poses[-1])
    np.testing.assert_allclose(poses[-1][:3, 3], tr.positions[-1])
