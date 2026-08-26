"""Koşu klasöründen tek çıktı videosu: kare-kare iz + anlık yön + H.264.

VO (ileri/geri/sağ/sol) trajectory.json'dan gelir; görselleştirme ondan
bağımsızdır — her kare Lucas-Kanade ile izlenir, iz gerçek ekran yoludur.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dcma.viz.plot import write_trajectory_plot
from dcma.viz.tracks import PointTracker

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TRAIL_SECONDS = 1.25

MIN_SPAN_M = 0.25
# Noktalar kenara yakın dursun, değmesin (kare genişliğinin %8'i pay).
EDGE_PAD = 0.08
SPAN_MARGIN = 1.0 / (1.0 - 2.0 * EDGE_PAD)


def minimap_view(path_xy: np.ndarray,
                 min_span: float = MIN_SPAN_M) -> tuple[np.ndarray, float]:
    """Kamera yalnızca yeşil–kırmızı mesafesine bağlı.

    Mesafe rekoru kırılınca (daha uzağa gidilince) kayar ve ölçeklenir.
    Mesafe azalınca (geri dönüş) son rekor görüşü donar — zoom yok.
    """
    xy = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros(2, dtype=np.float64), min_span
    green = xy[0]
    dist = np.linalg.norm(xy - green, axis=1)
    k = int(np.flatnonzero(dist == dist.max())[-1])
    anchor = xy[k]
    center = 0.5 * (green + anchor)
    extent = float(np.max(np.abs(np.stack([green, anchor]) - center)))
    return center, max(min_span, SPAN_MARGIN * extent)


def minimap_span(path_xy: np.ndarray, min_span: float = MIN_SPAN_M) -> float:
    return minimap_view(path_xy, min_span)[1]


DIR_COLORS = {
    "İLERİ": (0, 255, 0),
    "GERİ": (0, 0, 255),
    "SAĞA": (255, 0, 0),
    "SOLA": (255, 255, 0),
    "YUKARI": (255, 0, 255),
    "AŞAĞI": (0, 255, 255),
    "DURAGAN": (220, 220, 220),
}


def _direction(step: dict[str, Any] | None) -> str:
    if not step:
        return "DURAGAN"
    vals = {
        "İLERİ": float(step["forward"]),
        "GERİ": -float(step["forward"]),
        "SAĞA": float(step["right"]),
        "SOLA": -float(step["right"]),
        "YUKARI": float(step["up"]),
        "AŞAĞI": -float(step["up"]),
    }
    name = max(vals, key=vals.get)
    if vals[name] < 0.015:
        return "DURAGAN"
    return name


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


class FfmpegWriter:
    """libx264 + yuv420p: oynatıcıda gerçek FPS bozulmaz (OpenCV mp4v bozar)."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg bulunamadı")
        self.path = path
        w, h = width - (width % 2), height - (height % 2)
        self.width, self.height = w, h
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", f"{fps:.6f}",
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin açılamadı")
        self.stdin: BinaryIO = self._proc.stdin

    def write(self, frame: np.ndarray) -> None:
        img = frame
        h, w = img.shape[:2]
        if (w, h) != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def release(self) -> Path:
        try:
            self.stdin.close()
        except BrokenPipeError:
            pass
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg çıktı kodu {code}")
        return self.path


def _draw_badge(bgr: np.ndarray, text: str, color_bgr: tuple[int, int, int]) -> None:
    h, w = bgr.shape[:2]
    size = max(26, int(w * 0.10))
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (w, h)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 16, 12
    x1 = w - tw - pad_x * 2 - 16
    y1 = h - th - pad_y * 2 - 16
    x2, y2 = w - 16, h - 16
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, bgr, 0.28, 0, bgr)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color_bgr, 2)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((x1 + pad_x, y1 + pad_y - bbox[1]), text, font=font, fill=color_rgb)
    bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_minimap(frame: np.ndarray, path_xy: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    mw = min(140, max(96, w // 3))
    mh = mw
    x0, y0 = 10, h - mh - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (200, 200, 200), 1)
    if path_xy is None or len(path_xy) < 1:
        return
    xy = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    center, span = minimap_view(xy)

    def to_px(pt: np.ndarray) -> tuple[int, int]:
        rel = pt - center
        u = x0 + int((rel[0] / (2 * span) + 0.5) * (mw - 10)) + 5
        v = y0 + mh - 5 - int((rel[1] / (2 * span) + 0.5) * (mh - 10))
        return u, v

    pts = [to_px(p) for p in xy]
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.circle(frame, pts[0], 4, (0, 220, 0), -1)
    cv2.circle(frame, pts[-1], 4, (0, 0, 220), -1)


def draw_overlay(
    frame: np.ndarray,
    *,
    odo: dict[str, float],
    step: dict[str, Any] | None,
    path_xy: np.ndarray,
    frame_idx: int,
    tracker: PointTracker | None = None,
) -> np.ndarray:
    out = frame.copy()
    if tracker is not None:
        tracker.draw(out)
    direction = _direction(step)
    color = DIR_COLORS[direction]

    hud = [
        f"kare {frame_idx}",
        f"ileri  {odo['forward']:+.2f} m",
        f"saga   {odo['right']:+.2f} m",
        f"yukari {odo['up']:+.2f} m",
    ]
    y = 26
    for text in hud:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

    _draw_minimap(out, path_xy, color)
    _draw_badge(out, direction, color)
    return out


def _state_at_frame(payload: dict[str, Any], index: int) -> tuple[dict[str, float], dict[str, Any] | None, np.ndarray]:
    steps = payload["steps"]
    odo = {"forward": 0.0, "right": 0.0, "up": 0.0}
    step = None
    n_done = 0
    for s in steps:
        if s.get("frame_to") is None or int(s["frame_to"]) > index:
            if s.get("frame_from") is not None and int(s["frame_from"]) <= index:
                step = s
            break
        odo["forward"] += float(s["forward"])
        odo["right"] += float(s["right"])
        odo["up"] += float(s["up"])
        step = s
        n_done += 1
    pos = np.asarray(payload["positions"], dtype=np.float64)
    upto = min(n_done + 1, len(pos))
    path_xy = np.stack([pos[:upto, 0], pos[:upto, 2]], axis=1) if upto else np.zeros((1, 2))
    return odo, step, path_xy


def write_annotated_video(
    payload: dict[str, Any],
    frames_dir: str | Path,
    dest: str | Path,
) -> Path:
    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"PNG kare bulunamadı: {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"kare okunamadı: {frames[0]}")
    height, width = first.shape[:2]
    fps = float((payload.get("manifest") or {}).get("source_fps") or 30.0)
    if fps <= 0:
        fps = 30.0

    trail_len = max(8, int(round(fps * TRAIL_SECONDS)))
    tracker = PointTracker(max_points=70, trail_len=trail_len, min_distance=12)
    dest = Path(dest).with_suffix(".mp4")
    writer = FfmpegWriter(dest, fps, width, height)
    preview_idx = max(0, len(frames) // 2)
    preview_frame = None
    try:
        for idx, path in enumerate(frames):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tracker.update(gray)
            odo, step, path_xy = _state_at_frame(payload, idx)
            annotated = draw_overlay(
                frame, odo=odo, step=step, path_xy=path_xy,
                frame_idx=idx, tracker=tracker,
            )
            writer.write(annotated)
            if idx == preview_idx:
                preview_frame = annotated
    finally:
        written = writer.release()

    if preview_frame is not None:
        cv2.imwrite(str(written.with_name("preview.png")), preview_frame)
    return written


def write_outputs(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    frames_dir = Path((payload.get("manifest") or {}).get("frames_dir") or (run_dir / "frames"))
    plot_path = write_trajectory_plot(payload, run_dir / "plot.png")
    inner_video = write_annotated_video(payload, frames_dir, run_dir / "annotated.mp4")
    primary = run_dir.parent / f"{run_dir.name}.mp4"
    shutil.copy2(inner_video, primary)
    print(f"cikti videosu : {primary}")
    print(f"grafik        : {plot_path}")
    return {"video": primary, "inner_video": inner_video, "plot": plot_path,
            "preview": run_dir / "preview.png"}


def main() -> None:
    p = argparse.ArgumentParser(description="Mevcut DCMA koşusundan çıktı videosu üret")
    p.add_argument("--run", required=True, help="Result/<ad> klasörü (trajectory.json içerir)")
    args = p.parse_args()
    write_outputs(args.run)


if __name__ == "__main__":
    main()
