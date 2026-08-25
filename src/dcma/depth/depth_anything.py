"""Depth-Anything-V2 metrik checkpoint'leri (iç/dış mekân)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dcma.depth.backend import DepthBackend, ensure_size

CHECKPOINTS: dict[tuple[str, str], str] = {
    ("indoor", "base"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    ("indoor", "large"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    ("outdoor", "base"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    ("outdoor", "large"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


class DepthAnythingMetric(DepthBackend):
    def __init__(self, scene: str = "indoor", size: str = "large",
                 device: int = 0, cache_dir: str | Path | None = None,
                 lazy: bool = False) -> None:
        super().__init__(cache_dir=cache_dir)
        if scene == "auto":
            raise NotImplementedError(
                "sahne otomatik tespiti Faz 5'te eklenecek; "
                "şimdilik --scene indoor veya --scene outdoor verin")
        if (scene, size) not in CHECKPOINTS:
            raise ValueError(
                f"bilinmeyen sahne/boyut: {scene}/{size}; "
                f"geçerli olanlar: {sorted(CHECKPOINTS)}")
        self.scene = scene
        self.size = size
        self.checkpoint = CHECKPOINTS[(scene, size)]
        self.device = device
        self._pipe = None
        if not lazy:
            self._load()

    def _load(self) -> None:
        from transformers import pipeline
        self._pipe = pipeline("depth-estimation",
                              model=self.checkpoint, device=self.device)

    def predict(self, image: Image.Image) -> np.ndarray:
        if self._pipe is None:
            self._load()
        out = self._pipe(image.convert("RGB"))
        raw = out["predicted_depth"]
        arr = (raw.squeeze().float().cpu().numpy()
               if hasattr(raw, "cpu") else np.asarray(raw, dtype=np.float32))
        return ensure_size(arr, image.size)
