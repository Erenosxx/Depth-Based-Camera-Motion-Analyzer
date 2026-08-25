"""ORB özellik tespiti ve eşleme.

Faz 5'te LightGlue+SuperPoint ile değiştirilebilir; arayüz aynı kalacak şekilde
tasarlandı: iki gri görüntü al, eşleşen piksel koordinat çiftleri döndür.
"""

from __future__ import annotations

import cv2
import numpy as np

EMPTY = np.empty((0, 2), dtype=np.float32)


def detect_and_match(gray1: np.ndarray, gray2: np.ndarray,
                     n_features: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Eşleşen nokta çiftlerini (p1, p2) olarak döndürür; her biri (N, 2) float32."""
    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return EMPTY.copy(), EMPTY.copy()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    if not matches:
        return EMPTY.copy(), EMPTY.copy()

    p1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=np.float32)
    p2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=np.float32)
    return p1, p2
