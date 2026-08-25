"""Normalizasyon kaydı: kaynak videonun ne olduğu ve ona ne yapıldığı.

Manifest'in görevi, üretilen karelerin hangi dönüşümlerden geçtiğini ve
dolayısıyla hangi K matrisinin geçerli olduğunu kayıt altına almaktır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dcma.calib.intrinsics import Intrinsics


@dataclass
class VideoManifest:
    source_path: str
    source_width: int
    source_height: int
    source_fps: float
    source_codec: str
    rotation_tag_deg: int
    rotation_applied: bool
    frames_dir: str
    frame_count: int
    frame_width: int
    frame_height: int
    frame_format: str
    transforms: list[dict[str, Any]] = field(default_factory=list)
    intrinsics: Intrinsics | None = None

    def frame_seconds(self, index: int) -> float:
        return index / self.source_fps

    def stride_for_interval(self, seconds: float) -> int:
        """Verilen zaman aralığına karşılık gelen kare adımı (en az 1)."""
        return max(1, int(round(seconds * self.source_fps)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_path": self.source_path,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_fps": self.source_fps,
            "source_codec": self.source_codec,
            "rotation_tag_deg": self.rotation_tag_deg,
            "rotation_applied": self.rotation_applied,
            "frames_dir": self.frames_dir,
            "frame_count": self.frame_count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_format": self.frame_format,
            "transforms": self.transforms,
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
        }
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VideoManifest":
        payload = dict(d)
        intr = payload.pop("intrinsics", None)
        return cls(intrinsics=Intrinsics.from_dict(intr) if intr else None, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "VideoManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
