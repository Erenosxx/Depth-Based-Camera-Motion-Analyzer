"""cv2 ile ffmpeg arasındaki oryantasyon farkını ölçer ve kapatır.

Videolarda 'rotate' etiketi bulunabilir. ffmpeg bunu varsayılan olarak uygular.
OpenCV'nin uygulayıp uygulamadığı sürüme ve CAP_PROP_ORIENTATION_AUTO değerine
bağlıdır. Bu modül davranışı varsaymaz: ölçer, sonra gerekiyorsa elle döndürür.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dcma.video.normalize import probe_video


def cv2_applies_rotation(video_path: str | Path) -> bool:
    """cv2'nin dönme etiketini kendiliğinden uygulayıp uygulamadığını ölçer.

    Kare olmayan videoda boyut karşılaştırması kesin cevap verir. Kare (1:1)
    videoda boyut ipucu vermez; bu durumda OpenCV'nin bildirdiği ayar okunur.
    """
    info = probe_video(video_path)
    if info.rotation_deg not in (90, 270):
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video açılamadı: {video_path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if info.width != info.height:
            # donme uygulanmissa en/boy takla atmis olur
            return (w, h) == (info.height, info.width)
        prop = cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", -1))
        return bool(prop) if prop not in (-1, 0.0) else False
    finally:
        cap.release()


def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return frame
    if degrees % 360 == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees % 360 == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees % 360 == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"yalnızca 90'ın katları destekleniyor: {degrees}")


def read_frame_upright(video_path: str | Path, index: int) -> np.ndarray | None:
    """Videodan tek kare okur ve ffmpeg'in autorotate çıktısıyla aynı hale getirir."""
    info = probe_video(video_path)
    already = cv2_applies_rotation(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video açılamadı: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            return None
    finally:
        cap.release()

    if info.rotation_deg and not already:
        frame = _rotate(frame, info.rotation_deg)
    return frame
