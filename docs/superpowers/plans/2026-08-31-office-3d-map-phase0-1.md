# Ofis 3D haritası — Faz 0 + Faz 1 Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mevcut bir DCMA koşusundaki derinlik önbelleği + `T_wc` pozlarından, yeni model çalıştırmadan, metre ölçekli renkli `map.ply` üretmek.

**Architecture:** Çevrimdışı `python -m dcma.map.build --run Result/<ad>`. `trajectory.json` içindeki `steps[k].frame_from` → `poses[k]` eşlemesiyle her keyframe derinliği dünya çerçevesine unproject edilir, voxel ile seyreltilir, ASCII PLY yazılır. VO CLI’sine dokunulmaz.

**Tech Stack:** Mevcut `dcma` env (numpy, OpenCV, pytest). GPU yok. Yeni pip paketi yok.

**Spec:** [`docs/superpowers/specs/2026-08-31-office-3d-map-design.md`](../specs/2026-08-31-office-3d-map-design.md)

---

## Ortam — okumadan başlamayın

```bash
cd <repo-koku>
./Scripts/dcma.sh -c "import sys; print(sys.prefix)"   # dcma env
./Scripts/dcma.sh -m pytest tests/ -v
```

`LLM_training`’e paket kurulmaz. `AI_3D_env` import edilmez.

Test komutu (planın her yerinde):

```bash
./Scripts/dcma.sh -m pytest tests/test_fuse.py tests/test_ply.py tests/test_map_poses.py tests/test_map_build.py -v
```

---

## Dosya yapısı

| Dosya | Sorumluluk |
|---|---|
| `src/dcma/map/fuse.py` | ızgara unproject, `T_wc`, voxel |
| `src/dcma/map/ply.py` | ASCII PLY yaz/oku (test için oku) |
| `src/dcma/map/poses.py` | `trajectory.json` → `frame_idx → T_wc` |
| `src/dcma/map/build.py` | CLI: koşu dizininden `map.ply` |
| `tests/test_fuse.py` | sentetik koridor + iki kamera |
| `tests/test_ply.py` | PLY yuvarlak-gidiş |
| `tests/test_map_poses.py` | `frame_from` ↔ `poses[k]` kilidi |
| `tests/test_map_build.py` | sahte `Result/` entegrasyonu |
| `README.md` | harita kullanımı + sınırlamalar |

`occupancy.py` ve `cli.py` **değişmez** (Faz 0–1).

---

### Task 1: Dünya unproject + voxel

**Files:**
- Create: `src/dcma/map/fuse.py`
- Test: `tests/test_fuse.py`

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_fuse.py`:

```python
"""Sentetik koridor: duvarlar dünya x=±1 m; ikinci kamera +2 m ileri."""

from __future__ import annotations

import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.map.fuse import unproject_frame, voxel_downsample, fuse_frames


def _corridor_depth(K: Intrinsics, wall_x: float = 1.0) -> np.ndarray:
    depth = np.full((K.height, K.width), np.nan, dtype=np.float32)
    us = np.arange(K.width, dtype=np.float64)
    left, right = us < K.cx, us > K.cx
    depth[:, left] = ((-wall_x) * K.fx / (us[left] - K.cx))[np.newaxis, :]
    depth[:, right] = (wall_x * K.fx / (us[right] - K.cx))[np.newaxis, :]
    return depth


def _gray_rgb(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_identity_unproject_hits_walls_at_plus_minus_one_metre():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    xyz, rgb = unproject_frame(
        _corridor_depth(K), _gray_rgb(K.height, K.width), K, np.eye(4),
        stride=4, min_depth=0.3, max_depth=15.0)
    assert xyz.shape[1] == 3 and rgb.shape == (len(xyz), 3)
    xs = xyz[:, 0]
    left, right = xyz[xs < 0], xyz[xs > 0]
    assert len(left) >= 8 and len(right) >= 8
    assert left[:, 0].mean() == pytest.approx(-1.0, abs=0.2)
    assert right[:, 0].mean() == pytest.approx(1.0, abs=0.2)


def test_second_camera_extends_cloud_forward():
    K = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    depth = _corridor_depth(K)
    rgb = _gray_rgb(K.height, K.width)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[2, 3] = 2.0
    fused = fuse_frames(
        [(depth, rgb, T1), (depth, rgb, T2)],
        K, stride=4, min_depth=0.3, max_depth=15.0, voxel=0.10)
    first, _ = unproject_frame(depth, rgb, K, T1, stride=4)
    assert fused.xyz[:, 2].max() > first[:, 2].max() + 1.0


def test_voxel_downsample_reduces_count_and_keeps_cell():
    xyz = np.array([[0.01, 0.0, 1.0], [0.02, 0.0, 1.0], [1.0, 0.0, 1.0]],
                   dtype=np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    out_xyz, out_rgb = voxel_downsample(xyz, rgb, 0.10)
    assert len(out_xyz) == 2
    assert out_xyz.shape == out_rgb.shape
```

- [ ] **Step 2: Testi çalıştır — FAIL**

```bash
./Scripts/dcma.sh -m pytest tests/test_fuse.py -v
```

Beklenen: `ModuleNotFoundError: dcma.map.fuse`

- [ ] **Step 3: Minimal uygulama**

`src/dcma/map/fuse.py`:

```python
"""Derinlik + RGB + T_wc → dünya nokta bulutu; voxel seyreltme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcma.calib.intrinsics import Intrinsics

DEFAULT_STRIDE = 4


@dataclass
class FusedCloud:
    xyz: np.ndarray  # (N, 3) float64 dünya
    rgb: np.ndarray  # (N, 3) uint8


def unproject_frame(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: Intrinsics,
    T_wc: np.ndarray,
    stride: int = DEFAULT_STRIDE,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    rgb = np.asarray(rgb)
    h, w = depth.shape[:2]
    vs = np.arange(0, h, int(stride))
    us = np.arange(0, w, int(stride))
    uu, vv = np.meshgrid(us, vs)
    u = uu.ravel().astype(np.float64)
    v = vv.ravel().astype(np.float64)
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    z = depth[vi, ui].astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    valid = (
        np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
        & (z > min_depth) & (z < max_depth)
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    pts = np.stack([x[valid], y[valid], z[valid], np.ones(int(valid.sum()))])
    world = (np.asarray(T_wc, dtype=np.float64) @ pts)[:3].T
    color = rgb[vi[valid], ui[valid]].astype(np.uint8)
    return world, color


def voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if len(xyz) == 0:
        return xyz.reshape(0, 3), rgb.reshape(0, 3)
    if voxel <= 0:
        raise ValueError(f"voxel pozitif olmalı: {voxel}")
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], rgb[idx]


def fuse_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    K: Intrinsics,
    stride: int = DEFAULT_STRIDE,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
    voxel: float = 0.03,
) -> FusedCloud:
    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    for depth, rgb, T_wc in frames:
        xyz, color = unproject_frame(
            depth, rgb, K, T_wc, stride=stride,
            min_depth=min_depth, max_depth=max_depth)
        if len(xyz):
            chunks_xyz.append(xyz)
            chunks_rgb.append(color)
    if not chunks_xyz:
        return FusedCloud(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8))
    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    return FusedCloud(xyz=xyz, rgb=rgb)
```

`src/dcma/map/__init__.py` zaten occupancy için varsa dokunma.

- [ ] **Step 4: Testi çalıştır — PASS**

```bash
./Scripts/dcma.sh -m pytest tests/test_fuse.py -v
```

Beklenen: 3 passed

- [ ] **Step 5: Commit** (yalnızca kullanıcı isterse)

```bash
git add src/dcma/map/fuse.py tests/test_fuse.py
git commit -m "$(cat <<'EOF'
feat: dünya çerçevesinde derinlik birleştirme (voxel)

Ofis haritasının geometri çekirdeği; VO ve GPU yok.
EOF
)"
```

---

### Task 2: ASCII PLY

**Files:**
- Create: `src/dcma/map/ply.py`
- Test: `tests/test_ply.py`

- [ ] **Step 1: Başarısız testi yaz**

```python
from __future__ import annotations

import numpy as np

from dcma.map.ply import read_ply_xyz_rgb, write_ply_xyz_rgb


def test_ply_roundtrip(tmp_path):
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, -0.5, 2.0]], dtype=np.float64)
    rgb = np.array([[10, 20, 30], [255, 0, 1]], dtype=np.uint8)
    path = tmp_path / "map.ply"
    write_ply_xyz_rgb(path, xyz, rgb)
    text = path.read_text(encoding="ascii")
    assert "element vertex 2" in text
    assert "property uchar red" in text
    xyz2, rgb2 = read_ply_xyz_rgb(path)
    np.testing.assert_allclose(xyz2, xyz)
    np.testing.assert_array_equal(rgb2, rgb)
```

- [ ] **Step 2: FAIL** — `dcma.map.ply` yok

- [ ] **Step 3: Uygulama**

`src/dcma/map/ply.py`:

```python
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
```

- [ ] **Step 4: PASS**

```bash
./Scripts/dcma.sh -m pytest tests/test_ply.py -v
```

- [ ] **Step 5: Commit** (kullanıcı isterse)

---

### Task 3: `frame_from` → `T_wc` kilidi

**Files:**
- Create: `src/dcma/map/poses.py`
- Test: `tests/test_map_poses.py`

Bu test off-by-one’u kilitler. `cli.py` `splat(..., poses[-1], frame=i)` sonra `add_step(frame_from=i)`.

- [ ] **Step 1: Test**

```python
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
        pose_for_frame(table, 3)  # ara kare yok


def test_wrong_index_would_be_next_pose_not_identity():
    tr = Trajectory()
    t = np.array([0.1, 0.0, 0.0])
    tr.add_step(np.eye(3), t, frame_from=0, frame_to=4)
    payload = tr.to_dict()
    table = poses_from_trajectory(payload)
    # Hatalı: poses[1]'i frame 0'a vermek dünya kaydırır
    assert not np.allclose(pose_for_frame(table, 0), tr.poses[1])
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Uygulama**

`src/dcma/map/poses.py`:

```python
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
```

- [ ] **Step 4: PASS**

```bash
./Scripts/dcma.sh -m pytest tests/test_map_poses.py -v
```

- [ ] **Step 5: Commit** (kullanıcı isterse)

---

### Task 4: `dcma.map.build` — sahte Result entegrasyonu

**Files:**
- Create: `src/dcma/map/build.py`
- Test: `tests/test_map_build.py`

- [ ] **Step 1: Test** — sahte koşu dizini (2 PNG, 2 npy, mini trajectory)

```python
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from dcma.calib.intrinsics import Intrinsics
from dcma.map.build import build_map
from dcma.map.ply import read_ply_xyz_rgb
from dcma.vo.trajectory import Trajectory


def test_build_map_from_run_dir(tmp_path):
    K = Intrinsics(fx=100.0, fy=100.0, cx=4.5, cy=3.5, width=10, height=8)
    frames = tmp_path / "frames"
    cache = tmp_path / "depth_cache"
    frames.mkdir()
    cache.mkdir()
    rgb = np.full((8, 10, 3), 40, dtype=np.uint8)
    Image.fromarray(rgb).save(frames / "000000.png")
    Image.fromarray(rgb).save(frames / "000003.png")
    depth = np.full((8, 10), 2.0, dtype=np.float16)
    np.save(cache / "000000.npy", depth)
    np.save(cache / "000003.npy", depth)

    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, 0.5]), frame_from=0, frame_to=3)
    payload = tr.to_dict()
    payload["manifest"] = {
        "source_path": "x.mp4",
        "source_width": 10,
        "source_height": 8,
        "source_fps": 30.0,
        "source_codec": "h264",
        "rotation_tag_deg": 0,
        "rotation_applied": False,
        "frames_dir": str(frames),
        "frame_count": 2,
        "frame_width": 10,
        "frame_height": 8,
        "frame_format": "png",
        "transforms": [],
        "intrinsics": K.to_dict(),
    }
    (tmp_path / "trajectory.json").write_text(
        json.dumps(payload), encoding="utf-8")

    out = build_map(tmp_path, voxel=0.05, stride=2)
    assert out["ply"].is_file()
    xyz, rgb_out = read_ply_xyz_rgb(out["ply"])
    assert len(xyz) > 0
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))
    assert meta["n_points"] == len(xyz)
    assert meta["n_frames"] == 2
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Uygulama**

`src/dcma/map/build.py`:

```python
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

DEPTH_RANGE_M = {
    "indoor": (0.3, 15.0),
    "outdoor": (1.0, 60.0),
}


def _frame_png(run: Path, frames_dir: str, idx: int) -> Path:
    named = run / "frames" / f"{idx:06d}.png"
    if named.is_file():
        return named
    alt = Path(frames_dir) / f"{idx:06d}.png"
    return alt


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

    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    skipped: list[dict[str, Any]] = []
    used: list[int] = []
    for idx in sorted(table):
        png = _frame_png(run, frames_dir, idx)
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
    return {"ply": ply, "meta": meta_path, "n_points": meta["n_points"]}


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
    print(f"yazıldı {out['ply']}  nokta={out['n_points']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: PASS**

```bash
./Scripts/dcma.sh -m pytest tests/test_map_build.py tests/test_fuse.py tests/test_ply.py tests/test_map_poses.py -v
```

- [ ] **Step 5: Commit** (kullanıcı isterse)

---

### Task 5: README + gerçek koşu (gözlem, eşik yok)

**Files:**
- Modify: `README.md` — yol haritasına 3D harita; kullanım; sınırlamalar

- [ ] **Step 1:** README’ye ekle:

```bash
./Scripts/dcma.sh -m dcma.map.build --run Result/<ad> --voxel 0.03
# çıktı: Result/<ad>/map.ply
```

Sınırlama maddesi: harita VO drift’ini miras alır; döngü kapatma / VGGT Faz 2.

- [ ] **Step 2:** Mevcut ofis videosunun `Result/`’ı varsa (yeniden VO yok):

```bash
./Scripts/dcma.sh -m dcma.map.build --run Result/<mevcut_ad> --voxel 0.03
```

CloudCompare veya Blender’da `map.ply` aç. Beklenen: ilk bakış açısında duvar/mobilya; yandan bakınca videoda görülen yüzeyler dolu; hiç bakılmayan köşe boş.

- [ ] **Step 3:** Çift duvar / kayma varsa spec §6’ya not; VGGT’ye geçmeden **sayı yazma** (TUM yok). README “Bilinen Sınırlamalar”a bir cümle.

- [ ] **Step 4:** Commit (kullanıcı isterse)

```bash
git commit -m "docs: video koşusundan map.ply (Faz 1)"
```

---

## Faz 0–1 bittiğinde

Çalışan yazılım: `map.ply`. VGGT, mesh, döngü kapatma **yok** — bunlar spec §5 Faz 2–4, ayrı plan.

Handoff: kullanıcı spec + bu planı onaylamadan kod yazılmaz.
