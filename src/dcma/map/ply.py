"""ASCII PLY (xyz + rgb). Texture yok."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply_xyz_rgb(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[1] != 3 or rgb.shape[1] != 3:
        raise ValueError(f"şekil uyuşmaz: xyz={xyz.shape} rgb={rgb.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for i in range(n):
        x, y, z = xyz[i]
        r, g, b = (int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def read_ply_xyz_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_text(encoding="ascii").splitlines()
    i = raw.index("end_header")
    body = raw[i + 1 :]
    xyz, rgb = [], []
    for line in body:
        if not line.strip():
            continue
        p = line.split()
        xyz.append((float(p[0]), float(p[1]), float(p[2])))
        rgb.append((int(p[3]), int(p[4]), int(p[5])))
    return (
        np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
    )
