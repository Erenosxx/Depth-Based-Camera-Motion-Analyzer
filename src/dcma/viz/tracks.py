"""Kare-kare Lucas-Kanade izleme: her noktanın ekran yörüngesi ve rengi."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


def color_for_id(tid: int) -> tuple[int, int, int]:
    """Kararlı, ID'ye bağlı BGR renk — sarı tek renk değil."""
    hue = int((tid * 37) % 180)
    sat = int(160 + (tid * 17) % 95)
    val = int(210 + (tid * 11) % 45)
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class PointTracker:
    max_points: int = 70
    trail_len: int = 75
    min_distance: int = 14
    prev_gray: np.ndarray | None = None
    pts: np.ndarray | None = None
    ids: list[int] = field(default_factory=list)
    trails: list[deque[tuple[float, float]]] = field(default_factory=list)
    next_id: int = 1

    def seed_at(self, gray: np.ndarray, points_xy: np.ndarray) -> None:
        """Test ve ilk kare için bilinen noktalarla başlat."""
        self.prev_gray = gray.copy()
        self.pts = np.ascontiguousarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        self.ids = []
        self.trails = []
        for x, y in self.pts.reshape(-1, 2):
            self.ids.append(self.next_id)
            self.trails.append(deque([(float(x), float(y))], maxlen=self.trail_len))
            self.next_id += 1

    def update(self, gray: np.ndarray) -> None:
        if gray.ndim != 2:
            raise ValueError("gri kare beklenir")
        if self.prev_gray is None or self.pts is None or len(self.pts) == 0:
            self._seed(gray)
            self.prev_gray = gray.copy()
            return

        nxt, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.pts, None, **LK)
        kept_pts: list[np.ndarray] = []
        kept_ids: list[int] = []
        kept_trails: list[deque[tuple[float, float]]] = []
        h, w = gray.shape[:2]
        if nxt is not None and status is not None:
            status = status.reshape(-1)
            err = np.zeros(len(status)) if err is None else err.reshape(-1)
            for i, ok in enumerate(status):
                x, y = float(nxt[i, 0, 0]), float(nxt[i, 0, 1])
                ox, oy = float(self.pts[i, 0, 0]), float(self.pts[i, 0, 1])
                if not ok or err[i] > 15.0:
                    continue
                if not (2.0 <= x < w - 2.0 and 2.0 <= y < h - 2.0):
                    continue
                if (x - ox) ** 2 + (y - oy) ** 2 > 48.0 ** 2:
                    continue
                trail = self.trails[i]
                trail.append((x, y))
                kept_pts.append([[x, y]])
                kept_ids.append(self.ids[i])
                kept_trails.append(trail)

        if kept_pts:
            self.pts = np.array(kept_pts, dtype=np.float32)
            self.ids = kept_ids
            self.trails = kept_trails
        else:
            self.pts = np.empty((0, 1, 2), dtype=np.float32)
            self.ids = []
            self.trails = []

        if len(self.ids) < max(8, int(self.max_points * 0.65)):
            self._refill(gray)
        self.prev_gray = gray.copy()

    def draw(self, bgr: np.ndarray) -> None:
        for tid, trail in zip(self.ids, self.trails):
            if not trail:
                continue
            color = color_for_id(tid)
            pts = np.array(trail, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
            x, y = int(round(trail[-1][0])), int(round(trail[-1][1]))
            cv2.circle(bgr, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(bgr, (x, y), 3, color, -1, cv2.LINE_AA)

    def _seed(self, gray: np.ndarray) -> None:
        found = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_points, qualityLevel=0.01,
            minDistance=self.min_distance, blockSize=7)
        if found is None:
            self.pts = np.empty((0, 1, 2), dtype=np.float32)
            self.ids, self.trails = [], []
            return
        self.seed_at(gray, found.reshape(-1, 2))

    def _refill(self, gray: np.ndarray) -> None:
        need = self.max_points - len(self.ids)
        if need <= 0:
            return
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x, y in (self.pts.reshape(-1, 2) if self.pts is not None and len(self.pts) else []):
            cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)
        found = cv2.goodFeaturesToTrack(
            gray, maxCorners=need, qualityLevel=0.02,
            minDistance=self.min_distance, blockSize=7, mask=mask)
        if found is None:
            return
        extra = found.reshape(-1, 2)
        old = self.pts.reshape(-1, 2) if self.pts is not None and len(self.pts) else np.empty((0, 2))
        merged = np.vstack([old, extra]) if len(old) else extra
        self.pts = merged.reshape(-1, 1, 2).astype(np.float32)
        for x, y in extra:
            self.ids.append(self.next_id)
            self.trails.append(deque([(float(x), float(y))], maxlen=self.trail_len))
            self.next_id += 1
