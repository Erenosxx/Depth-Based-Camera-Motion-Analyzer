from __future__ import annotations

import numpy as np
import pytest

from dcma.map.poses import pose_for_frame, poses_from_trajectory
from dcma.vo.trajectory import Trajectory


def test_frame_from_uses_pose_before_step():
    tr = Trajectory()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.2])
    tr.add_step(R, t, frame_from=0, frame_to=5)
    tr.add_step(R, t, frame_from=5, frame_to=10)
    payload = tr.to_dict()
    table = poses_from_trajectory(payload)
    T0 = pose_for_frame(table, 0)
    T5 = pose_for_frame(table, 5)
    np.testing.assert_allclose(T0, np.eye(4))
    np.testing.assert_allclose(T5, tr.poses[1])
    with pytest.raises(KeyError):
        pose_for_frame(table, 3)


def test_wrong_index_would_be_next_pose_not_identity():
    tr = Trajectory()
    t = np.array([0.1, 0.0, 0.0])
    tr.add_step(np.eye(3), t, frame_from=0, frame_to=4)
    payload = tr.to_dict()
    table = poses_from_trajectory(payload)
    assert not np.allclose(pose_for_frame(table, 0), tr.poses[1])
