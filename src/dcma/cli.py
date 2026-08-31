"""Uçtan uca çalıştırma: video -> normalize -> derinlik -> poz -> trajectory.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dcma.calib.intrinsics import Intrinsics
from dcma.depth.depth_anything import DepthAnythingMetric
from dcma.video.normalize import normalize_video, probe_video
from dcma.vo.features import detect_and_match
from dcma.vo.pose import (
    DEFAULT_YAW_SCALE,
    backproject,
    estimate_pose,
    maybe_inplace_yaw,
    maybe_lateral_strafe,
    scale_yaw,
)
from dcma.vo.trajectory import Trajectory
from dcma.viz.annotate import write_outputs
from dcma.map.occupancy import OccupancyGrid
from dcma.map.build import build_map
from dcma.map.fuse import DEFAULT_UP_MAX

# Gecerli derinlik araliklari sahneye gore degisir: ic mekan modeli ~20 m'ye,
# dis mekan modeli ~80 m'ye kadar egitildi. Araligin disi guvenilmez.
DEPTH_RANGE_M = {
    "indoor": (0.3, 15.0),
    "outdoor": (1.0, 60.0),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcma", description="Derinlik tabanlı metrik kamera hareket analizi")
    p.add_argument("--video", required=True, help="girdi videosu (herhangi format)")
    p.add_argument("--out", required=True, help="çıktı dizini")
    p.add_argument("--scene", required=True, choices=["indoor", "outdoor", "auto"],
                   help="sahne türü; 'auto' Faz 5'te gelecek")
    p.add_argument("--size", default="large", choices=["base", "large"])
    p.add_argument("--fov-long", dest="fov_long", type=float, default=70.0,
                   help="kalibrasyon yoksa kaynağın UZUN kenarı boyunca görüş "
                        "açısı (derece). Kamera spesifikasyonları bu eksene "
                        "göre verilir; portre/yatay ayrımından bağımsızdır.")
    p.add_argument("--interval", type=float, default=0.15,
                   help="kare çiftleri arası hedef süre (saniye)")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p.add_argument("--max-edge", dest="max_edge", type=int, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-map", dest="no_map", action="store_true",
                   help="3D harita (map.ply) üretme")
    p.add_argument("--voxel", type=float, default=0.03,
                   help="3D harita voxel boyutu (metre)")
    p.add_argument("--map-stride", dest="map_stride", type=int, default=4,
                   help="3D harita piksel adımı")
    p.add_argument("--yaw-scale", dest="yaw_scale", type=float,
                   default=DEFAULT_YAW_SCALE,
                   help="PnP yaw çarpanı (1.0 ham). Ofis L-köşesinde 0.88 varsayılan.")
    p.add_argument("--up-max", dest="up_max", type=float, default=DEFAULT_UP_MAX,
                   help="3D haritada tavan: dünya yukarı (m) bundan büyükse atılır "
                        "(ilk kare göz=0; ofis tavanı ~1.1)")
    return p


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(args.video)
    # Uzun kenar konvansiyonu: natif portre kaynakta from_fov açıyı kısa
    # eksene uygular, fx küçük çıkar, dönme ve yanal öteleme şişer.
    k_source = Intrinsics.from_fov_long_edge(
        info.width, info.height, args.fov_long)
    manifest = normalize_video(args.video, out_dir,
                               max_edge=args.max_edge,
                               max_frames=args.max_frames,
                               intrinsics=k_source)
    K = manifest.intrinsics
    if K is None:
        raise RuntimeError("manifest intrinsics içermiyor")

    frames = sorted(Path(manifest.frames_dir).glob("*.png"))
    stride = manifest.stride_for_interval(args.interval)
    print(f"kare sayısı={len(frames)}  adım={stride} "
          f"({args.interval}s @ {manifest.source_fps:.2f} fps)")

    backend = DepthAnythingMetric(scene=args.scene, size=args.size,
                                  device=args.device,
                                  cache_dir=out_dir / "depth_cache",
                                  lazy=True)
    min_depth, max_depth = DEPTH_RANGE_M[args.scene]
    traj = Trajectory()
    skipped: list[dict] = []
    occ = OccupancyGrid(min_depth=min_depth, max_depth=max_depth)

    for i in range(0, len(frames) - stride, stride):
        j = i + stride
        img_i = Image.open(frames[i]).convert("RGB")
        gray_i = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        gray_j = cv2.imread(str(frames[j]), cv2.IMREAD_GRAYSCALE)

        p1, p2 = detect_and_match(gray_i, gray_j)
        if len(p1) < 20:
            skipped.append({"from": i, "to": j, "reason": "yetersiz eşleşme",
                            "matches": int(len(p1))})
            continue

        depth_i = backend.predict_cached(img_i, key=f"{i:06d}")
        P3d, z = backproject(depth_i, K, p1)
        valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
        if valid.sum() < 20:
            skipped.append({"from": i, "to": j, "reason": "yetersiz geçerli derinlik",
                            "valid": int(valid.sum())})
            continue

        res = estimate_pose(P3d[valid], p2[valid], K)
        if res is None:
            skipped.append({"from": i, "to": j, "reason": "PnP çözülemedi"})
            continue

        R, t = maybe_inplace_yaw(res.R, res.t)
        R = scale_yaw(R, float(getattr(args, "yaw_scale", DEFAULT_YAW_SCALE)))
        R, t = maybe_lateral_strafe(R, t)
        occ.splat(depth_i, K, traj.poses[-1], frame_idx=i)
        traj.add_step(R, t, inliers=res.inliers,
                      reproj_err=res.reproj_err, frame_from=i, frame_to=j)

    payload = traj.to_dict()
    payload["skipped"] = skipped
    payload["manifest"] = manifest.to_dict()
    (out_dir / "trajectory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    occ.to_npz(out_dir / "occupancy.npz")
    occ.write_png(out_dir / "occupancy.png")

    odo = payload["odometer"]
    net = payload["net_displacement"]
    print(f"\nadım={payload['step_count']}  atlanan={len(skipped)}")
    print(f"odometre : ileri={odo['forward']:+.3f} m  sağa={odo['right']:+.3f} m  "
          f"yukarı={odo['up']:+.3f} m")
    print(f"net      : ileri={net['forward']:+.3f} m  sağa={net['right']:+.3f} m  "
          f"yukarı={net['up']:+.3f} m  (|{net['magnitude']:.3f}| m)")
    print(f"yol uzunluğu: {payload['path_length']:.3f} m")
    if not getattr(args, "no_map", False):
        mapped = build_map(
            out_dir,
            voxel=float(getattr(args, "voxel", 0.03)),
            stride=int(getattr(args, "map_stride", 4)),
            min_depth=min_depth,
            max_depth=max_depth,
            up_max=float(getattr(args, "up_max", DEFAULT_UP_MAX)),
        )
        print(f"3D harita   : {mapped['ply']}  ({mapped['n_points']} nokta)")
        print(f"3D önizleme : {mapped['preview']}")
    write_outputs(out_dir)
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
