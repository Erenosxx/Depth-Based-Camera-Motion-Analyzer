"""Herhangi bir formattaki videoyu kanonik kare dizisine çevirir.

Yapılanlar:
  - ffprobe ile geometri, fps, codec ve dönme etiketi okunur
  - ffmpeg ile dönme AÇIKÇA uygulanır (autorotate)
  - kareler PNG olarak yazılır (kayıpsız; JPG özellik tespitine gürültü katar)
  - konum/GPS metadata'sı taşınmaz (yalnızca görüntü verisi çıkarılır)
  - uygulanan her dönüşüm manifest'e kaydedilir ve K buna göre güncellenir
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dcma.calib.intrinsics import Intrinsics
from dcma.video.manifest import VideoManifest


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    codec: str
    rotation_deg: int
    nb_frames: int | None


def _require_tools() -> None:
    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} bulunamadı; ffmpeg kurulu olmalı")


def _parse_fps(text: str) -> float:
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        if den_f == 0.0:
            return 0.0
        return float(num) / den_f
    return float(text)


def _rotation_from_stream(stream: dict) -> int:
    """Dönme hem 'rotate' tag'inde hem de displaymatrix side_data'sında olabilir."""
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        return int(round(float(tags["rotate"]))) % 360
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            # displaymatrix rotation isareti ters konvansiyondadir
            return int(round(-float(sd["rotation"]))) % 360
    return 0


def probe_video(path: str | Path) -> VideoInfo:
    _require_tools()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams") or []
    if not streams:
        raise ValueError(f"videoda görüntü akışı bulunamadı: {path}")
    s = streams[0]
    nb = s.get("nb_frames")
    return VideoInfo(
        width=int(s["width"]),
        height=int(s["height"]),
        fps=_parse_fps(s.get("r_frame_rate") or "0/1"),
        codec=str(s.get("codec_name") or "unknown"),
        rotation_deg=_rotation_from_stream(s),
        nb_frames=int(nb) if nb not in (None, "N/A") else None,
    )


def normalize_video(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    max_edge: int | None = None,
    max_frames: int | None = None,
    intrinsics: Intrinsics | None = None,
) -> VideoManifest:
    """Videoyu PNG karelere çevirir ve manifest üretir."""
    _require_tools()
    info = probe_video(video_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    transforms: list[dict] = []

    # ffmpeg varsayilan olarak autorotate uygular; aciklikla kaydediyoruz.
    if info.rotation_deg:
        transforms.append({"kind": "autorotate", "degrees": info.rotation_deg})
        rotated_w, rotated_h = (
            (info.height, info.width) if info.rotation_deg in (90, 270)
            else (info.width, info.height))
    else:
        rotated_w, rotated_h = info.width, info.height

    k = intrinsics
    if k is not None and (k.width, k.height) != (info.width, info.height):
        raise ValueError(
            f"verilen intrinsics kaynak boyutuyla uyuşmuyor: "
            f"{k.width}x{k.height} != {info.width}x{info.height}")
    if k is not None and info.rotation_deg:
        k = k.rotated(info.rotation_deg)

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video_path)]
    vf: list[str] = []
    if max_edge is not None and max(rotated_w, rotated_h) > max_edge:
        scale = max_edge / max(rotated_w, rotated_h)
        target_w = int(round(rotated_w * scale))
        target_h = int(round(rotated_h * scale))
        vf.append(f"scale={target_w}:{target_h}:flags=lanczos")
        transforms.append({"kind": "scale", "factor": scale,
                           "width": target_w, "height": target_h})
        if k is not None:
            k = k.scaled_to_max_edge(max_edge)
        frame_w, frame_h = target_w, target_h
    else:
        frame_w, frame_h = rotated_w, rotated_h

    if vf:
        cmd += ["-vf", ",".join(vf)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-map_metadata", "-1", str(frames_dir / "frame_%06d.png")]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    written = sorted(frames_dir.glob("*.png"))
    if not written:
        raise RuntimeError("ffmpeg hiç kare üretmedi")

    manifest = VideoManifest(
        source_path=str(Path(video_path).resolve()),
        source_width=info.width,
        source_height=info.height,
        source_fps=info.fps,
        source_codec=info.codec,
        rotation_tag_deg=info.rotation_deg,
        rotation_applied=bool(info.rotation_deg),
        frames_dir=str(frames_dir.resolve()),
        frame_count=len(written),
        frame_width=frame_w,
        frame_height=frame_h,
        frame_format="png",
        transforms=transforms,
        intrinsics=k,
    )
    manifest.save(out_dir / "manifest.json")
    return manifest
