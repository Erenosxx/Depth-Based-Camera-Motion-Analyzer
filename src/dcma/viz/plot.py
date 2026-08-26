"""Yörünge grafiği: üstten görünüm + yükseklik profili."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_trajectory_plot(payload: dict[str, Any], path: str | Path,
                          occupancy: Any | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    pos = np.asarray(payload["positions"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise ValueError("positions (N, 3) olmalı")

    right, down, forward = pos[:, 0], pos[:, 1], pos[:, 2]
    up = -down

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    if occupancy is not None:
        counts, origin = occupancy.dense()
        res = float(occupancy.resolution)
        hh, ww = counts.shape
        if counts.max() > 0:
            extent = [origin[0], origin[0] + ww * res,
                      origin[1], origin[1] + hh * res]
            ax.imshow(counts, origin="lower", extent=extent, cmap="gray_r",
                      interpolation="nearest", alpha=0.85, aspect="equal",
                      vmin=0, vmax=1, zorder=0)
    ax.plot(right, forward, color="#1f77b4", linewidth=1.5, zorder=2)
    ax.scatter(right[0], forward[0], color="#2ca02c", s=36, zorder=3, label="başlangıç")
    ax.scatter(right[-1], forward[-1], color="#d62728", s=36, zorder=3, label="bitiş")
    ax.set_xlabel("sağa (m)")
    ax.set_ylabel("ileri (m)")
    ax.set_title("Üstten görünüm")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(np.arange(len(up)), up, color="#9467bd", linewidth=1.5)
    ax.set_xlabel("adım")
    ax.set_ylabel("yukarı (m)")
    ax.set_title("Yükseklik profili")
    ax.grid(True, alpha=0.3)

    net = payload.get("net_displacement") or {}
    path_len = payload.get("path_length")
    fig.suptitle(
        f"net |{net.get('magnitude', 0):.2f}| m   yol {path_len:.2f} m"
        if path_len is not None else "",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
