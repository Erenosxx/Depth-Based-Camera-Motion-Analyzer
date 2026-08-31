"""3D nokta bulutu önizlemesi (üstten + yandan). Texture yok."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_map_preview(
    xyz: np.ndarray,
    rgb: np.ndarray,
    path: str | Path,
    max_points: int = 25000,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(xyz) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("3D harita (boş)")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    colors = np.clip(rgb / 255.0, 0.0, 1.0)
    right, down, forward = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    up = -down

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(right, forward, up, c=colors, s=1.0, linewidths=0)
    ax.set_xlabel("sağa (m)")
    ax.set_ylabel("ileri (m)")
    ax.set_zlabel("yukarı (m)")
    ax.set_title("3D harita")
    ax.view_init(elev=18, azim=-70)
    try:
        ax.set_box_aspect((1, 1, 0.35))
    except Exception:
        pass

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(right, forward, c=colors, s=1.5, linewidths=0)
    ax2.set_xlabel("sağa (m)")
    ax2.set_ylabel("ileri (m)")
    ax2.set_title("Üstten (3D projeksiyon)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
