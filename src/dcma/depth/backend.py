"""Derinlik backend'leri için ortak arayüz.

predict(), METRE cinsinden float32 bir derinlik haritası döndürür ve haritanın
girdi görüntüsüyle aynı boyutta olmasını garanti eder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class DepthBackend(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """Girdi boyutunda, metre cinsinden float32 derinlik haritası."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def predict_cached(self, image: Image.Image, key: str) -> np.ndarray:
        """Önbellekli çıkarım. Aynı kare iki kez modele girmez."""
        if not self.cache_dir:
            return self.predict(image)
        path = self.cache_dir / f"{key}.npy"
        if path.is_file():
            return np.load(path).astype(np.float32)
        depth = self.predict(image)
        cached = depth.astype(np.float16)
        np.save(path, cached)
        return cached.astype(np.float32)


def ensure_size(depth: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Derinlik haritasını (genişlik, yükseklik) boyutuna getirir."""
    width, height = size_wh
    if depth.shape == (height, width):
        return depth.astype(np.float32)
    import cv2
    return cv2.resize(depth.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR)
