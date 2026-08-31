"""trajectory.json → frame_idx → T_wc (4×4)."""

from __future__ import annotations

from typing import Any

import numpy as np


def poses_from_trajectory(payload: dict[str, Any]) -> dict[int, np.ndarray]:
    poses = [np.asarray(T, dtype=np.float64) for T in payload["poses"]]
    steps = payload.get("steps") or []
    table: dict[int, np.ndarray] = {}
    if not steps:
        if poses:
            table[0] = poses[0]
        return table
    for k, step in enumerate(steps):
        src = step.get("frame_from")
        if src is None:
            continue
        table[int(src)] = poses[k]
        dst = step.get("frame_to")
        if dst is not None and k + 1 < len(poses):
            table[int(dst)] = poses[k + 1]
    return table


def pose_for_frame(table: dict[int, np.ndarray], frame_idx: int) -> np.ndarray:
    if frame_idx not in table:
        raise KeyError(f"kare {frame_idx} için T_wc yok")
    return table[frame_idx]
