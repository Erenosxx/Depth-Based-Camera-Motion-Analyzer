"""Mevcut DCMA koşusundan map.ply (VO tekrar çalışmaz)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dcma.calib.intrinsics import Intrinsics
from dcma.map.fuse import fuse_frames
from dcma.map.ply import write_ply_xyz_rgb
from dcma.map.poses import poses_from_trajectory
from dcma.map.preview import write_map_preview

DEPTH_RANGE_M = {
    "indoor": (0.3, 15.0),
    "outdoor": (1.0, 60.0),
}


def _list_frame_pngs(run: Path, frames_dir: str) -> list[Path]:
    """cli.py ile aynı: sorted glob. ffmpeg çıktısı frame_%06d.png (1-tabanlı)."""
    local = run / "frames"
    d = local if local.is_dir() else Path(frames_dir)
    return sorted(d.glob("*.png"))


def _frame_png(run: Path, frames_dir: str, idx: int,
               pngs: list[Path] | None = None) -> Path:
    named = run / "frames" / f"{idx:06d}.png"
    if named.is_file():
        return named
    alt = Path(frames_dir) / f"{idx:06d}.png"
    if alt.is_file():
        return alt
    if pngs is None:
        pngs = _list_frame_pngs(run, frames_dir)
    if 0 <= idx < len(pngs):
        return pngs[idx]
    return named


def _depth_npy(run: Path, idx: int) -> Path:
    return run / "depth_cache" / f"{idx:06d}.npy"


def build_map(
    run: Path,
    voxel: float = 0.03,
    stride: int = 4,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
) -> dict[str, Any]:
    run = Path(run)
    payload = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    K = Intrinsics.from_dict(manifest["intrinsics"])
    table = poses_from_trajectory(payload)
    frames_dir = str(manifest.get("frames_dir") or (run / "frames"))
    pngs = _list_frame_pngs(run, frames_dir)

    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    skipped: list[dict[str, Any]] = []
    used: list[int] = []
    for idx in sorted(table):
        png = _frame_png(run, frames_dir, idx, pngs=pngs)
        npy = _depth_npy(run, idx)
        if not png.is_file() or not npy.is_file():
            skipped.append({"frame": idx, "reason": "png veya derinlik yok"})
            continue
        rgb = np.asarray(Image.open(png).convert("RGB"))
        depth = np.load(npy).astype(np.float32)
        packed.append((depth, rgb, table[idx]))
        used.append(idx)

    if not packed:
        raise RuntimeError(f"birleştirilecek kare yok: {run}")

    cloud = fuse_frames(
        packed, K, stride=stride, min_depth=min_depth,
        max_depth=max_depth, voxel=voxel)
    ply = write_ply_xyz_rgb(run / "map.ply", cloud.xyz, cloud.rgb)
    preview = write_map_preview(cloud.xyz, cloud.rgb, run / "map_preview.png")
    meta = {
        "voxel": voxel,
        "stride": stride,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "n_points": int(len(cloud.xyz)),
        "n_frames": len(used),
        "frames": used,
        "skipped": skipped,
        "xyz_min": cloud.xyz.min(axis=0).tolist() if len(cloud.xyz) else [],
        "xyz_max": cloud.xyz.max(axis=0).tolist() if len(cloud.xyz) else [],
    }
    meta_path = run / "map_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return {"ply": ply, "meta": meta_path, "preview": preview,
            "n_points": meta["n_points"]}


def main() -> None:
    p = argparse.ArgumentParser(prog="dcma.map.build")
    p.add_argument("--run", required=True, help="Result/<ad> koşu dizini")
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor")
    args = p.parse_args()
    lo, hi = DEPTH_RANGE_M[args.scene]
    out = build_map(Path(args.run), voxel=args.voxel, stride=args.stride,
                    min_depth=lo, max_depth=hi)
    print(f"yazıldı {out['ply']}  nokta={out['n_points']}  önizleme={out['preview']}")


if __name__ == "__main__":
    main()
