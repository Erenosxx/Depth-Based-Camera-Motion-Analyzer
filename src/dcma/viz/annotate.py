"""Koşu klasöründen tek çıktı videosu üretir: nokta hareketi + anlık yön.

GPU / derinlik tekrar çalışmaz. frames/ yalnızca ara bellek;
asıl teslim Result/<ad>.mp4 dosyasıdır.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dcma.vo.features import detect_and_match
from dcma.viz.plot import write_trajectory_plot

MAX_POINTS = 80
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# BGR — eski 6 yön paleti
DIR_COLORS = {
    "İLERİ": (0, 255, 0),
    "GERİ": (0, 0, 255),
    "SAĞA": (255, 0, 0),
    "SOLA": (255, 255, 0),
    "YUKARI": (255, 0, 255),
    "AŞAĞI": (0, 255, 255),
    "DURAGAN": (220, 220, 220),
}

ARROW_COLOR = (0, 255, 255)
POINT_COLOR = (0, 220, 255)


@dataclass
class TrackSet:
    frame_from: int
    frame_to: int
    p1: np.ndarray
    p2: np.ndarray


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


def _open_writer(path: Path, fps: float, size_wh: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    width, height = size_wh
    candidates = [
        (path.with_suffix(".mp4"), "mp4v"),
        (path.with_suffix(".mp4"), "avc1"),
        (path.with_suffix(".avi"), "XVID"),
        (path.with_suffix(".avi"), "MJPG"),
    ]
    for dest, codec in candidates:
        writer = cv2.VideoWriter(
            str(dest), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, dest
        writer.release()
    raise RuntimeError("VideoWriter açılamadı; OpenCV codec desteği yok")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def _draw_badge(bgr: np.ndarray, text: str, color_bgr: tuple[int, int, int]) -> None:
    """Sağ altta büyük anlık yön etiketi (Türkçe, DejaVu)."""
    h, w = bgr.shape[:2]
    size = max(22, int(w * 0.09))
    font = _font(size)
    rgb = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(rgb)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 14, 10
    x1 = w - tw - pad_x * 2 - 12
    y1 = h - th - pad_y * 2 - 12
    x2, y2 = w - 12, h - 12
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0, bgr)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color_bgr, 2)

    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((x1 + pad_x, y1 + pad_y - bbox[1]), text, font=font, fill=color_rgb)
    bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_tracks(out: np.ndarray, p1: np.ndarray, p2: np.ndarray, alpha: float) -> None:
    if p1 is None or p2 is None or len(p1) == 0 or len(p1) != len(p2):
        return
    if len(p1) > MAX_POINTS:
        idx = np.linspace(0, len(p1) - 1, MAX_POINTS).astype(int)
        p1, p2 = p1[idx], p2[idx]
    cur = p1 + (p2 - p1) * float(np.clip(alpha, 0.0, 1.0))
    h, w = out.shape[:2]
    for start, cur_pt in zip(p1, cur):
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(cur_pt[0])), int(round(cur_pt[1]))
        if not (0 <= x1 < w and 0 <= y1 < h):
            continue
        if abs(x1 - x0) + abs(y1 - y0) >= 2:
            cv2.arrowedLine(out, (x0, y0), (x1, y1), ARROW_COLOR, 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(out, (x1, y1), 3, POINT_COLOR, -1, cv2.LINE_AA)


def _draw_minimap(frame: np.ndarray, path_xy: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    mw = min(130, max(90, w // 3))
    mh = mw
    x0, y0 = 8, h - mh - 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (180, 180, 180), 1)
    if path_xy is None or len(path_xy) < 1:
        return
    xy = np.asarray(path_xy, dtype=np.float64)
    span = max(float(np.max(np.abs(xy))), 0.25)

    def to_px(pt: np.ndarray) -> tuple[int, int]:
        u = x0 + int((pt[0] / (2 * span) + 0.5) * (mw - 8)) + 4
        v = y0 + mh - 4 - int((pt[1] / (2 * span) + 0.5) * (mh - 8))
        return u, v

    pts = [to_px(p) for p in xy]
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(frame, pts[0], 3, (0, 220, 0), -1)
    cv2.circle(frame, pts[-1], 3, (0, 0, 220), -1)


def draw_overlay(
    frame: np.ndarray,
    *,
    p1: np.ndarray | None = None,
    p2: np.ndarray | None = None,
    alpha: float = 0.0,
    points: np.ndarray | None = None,
    odo: dict[str, float],
    step: dict[str, Any] | None,
    path_xy: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    """Nokta okları + kümülatif metre + büyük anlık yön. BGR uint8."""
    out = frame.copy()
    direction = _direction(step)
    color = DIR_COLORS[direction]

    if p1 is None and points is not None:
        p1 = points
        p2 = points
    _draw_tracks(out, p1 if p1 is not None else np.empty((0, 2)),
                 p2 if p2 is not None else np.empty((0, 2)), alpha)

    hud = [
        f"kare {frame_idx}",
        f"ileri {odo['forward']:+.2f} m",
        f"saga  {odo['right']:+.2f} m",
        f"yukari {odo['up']:+.2f} m",
    ]
    y = 24
    for text in hud:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

    _draw_minimap(out, path_xy, color)
    _draw_badge(out, direction, color)
    return out


def _collect_tracks(frames: list[Path], steps: list[dict[str, Any]]) -> list[TrackSet]:
    tracks: list[TrackSet] = []
    for step in steps:
        i, j = int(step["frame_from"]), int(step["frame_to"])
        if i < 0 or j >= len(frames) or i >= j:
            continue
        gray_i = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        gray_j = cv2.imread(str(frames[j]), cv2.IMREAD_GRAYSCALE)
        if gray_i is None or gray_j is None:
            continue
        p1, p2 = detect_and_match(gray_i, gray_j)
        if len(p1) == 0:
            continue
        tracks.append(TrackSet(frame_from=i, frame_to=j, p1=p1, p2=p2))
    return tracks


def _track_at(tracks: list[TrackSet], index: int) -> tuple[np.ndarray, np.ndarray, float]:
    empty = np.empty((0, 2), dtype=np.float32)
    for t in tracks:
        if t.frame_from <= index <= t.frame_to:
            span = max(1, t.frame_to - t.frame_from)
            return t.p1, t.p2, (index - t.frame_from) / span
    return empty, empty, 0.0


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

    tracks = _collect_tracks(frames, payload.get("steps") or [])
    writer, written = _open_writer(Path(dest), fps, (width, height))
    preview_idx = max(0, len(frames) // 2)
    preview_frame = None
    try:
        for idx, path in enumerate(frames):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            odo, step, path_xy = _state_at_frame(payload, idx)
            p1, p2, alpha = _track_at(tracks, idx)
            annotated = draw_overlay(
                frame, p1=p1, p2=p2, alpha=alpha,
                odo=odo, step=step, path_xy=path_xy, frame_idx=idx,
            )
            writer.write(annotated)
            if idx == preview_idx:
                preview_frame = annotated
    finally:
        writer.release()

    if preview_frame is not None:
        cv2.imwrite(str(Path(written).with_name("preview.png")), preview_frame)
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
