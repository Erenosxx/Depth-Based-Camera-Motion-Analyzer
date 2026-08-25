# Metrik 6-DoF Görsel Odometri — Faz 0 + Faz 1 Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Herhangi bir formattaki iç mekân videosundan, metre cinsinden kare-kare kamera hareketi üreten ve sentetik testle doğrulanmış bir pipeline kurmak.

**Architecture:** Video önce kanonik karelere ve bir manifest'e normalize edilir (dönme uygulanır, her dönüşüm `K` güncellemesiyle birlikte kaydedilir). Her kare için metrik derinlik haritası üretilir. Eşleşen özellikler `K` ve derinlikle 3B'ye taşınır, `cv2.solvePnPRansac` ile kare çiftleri arası poz çözülür, pozlar birikirilerek yörünge oluşur. Doğruluk, derinlik modelinden bağımsız sentetik sahne testleriyle güvence altına alınır.

**Tech Stack:** Python 3.11, PyTorch 2.6 (cu124), transformers (`pipeline("depth-estimation")`), Depth-Anything-V2 metrik checkpoint'leri, OpenCV, NumPy, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-25-metric-6dof-vo-design.md`](../specs/2026-08-25-metric-6dof-vo-design.md)

---

## Ortam — okumadan başlamayın

Proje **`dcma`** conda ortamında geliştirilir. `LLM_training` ortamına **dokunulmaz**
(aktif Gemma eğitimi barındırıyor ve cuDNN yığını bozuk: `nvidia-cudnn-cu13`,
`nvidia-cudnn-cu12`'nin üzerine yazdığı için her `conv2d` çağrısı
`CUDNN_STATUS_NOT_INITIALIZED` veriyor).

**Tuzak:** Bu makinenin shell'inde `VIRTUAL_ENV=/path/to/envs/other_project`
set edilmiş durumda. `pip` bu değişkeni conda ortamına **tercih eder**, dolayısıyla
`conda run -n dcma pip install ...` paketleri sessizce `LLM_training`'e kurar.

Her python/pip çağrısında ortamın doğru olduğundan emin olmak için:

```bash
export DPY=/path/to/envs/dcma/bin/python
alias dpy='env -u VIRTUAL_ENV -u PYTHONPATH $DPY'
# dogrulama - her zaman /path/to/envs/dcma yazmali:
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -c "import sys; print(sys.prefix)"
```

Testleri çalıştırma biçimi (planın her yerinde bu kullanılır):

```bash
cd <repo-koku>
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/ -v
```

---

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `pyproject.toml` | pytest yapılandırması (`pythonpath = ["src"]`), proje metadata'sı |
| `src/dcma/__init__.py` | paket kökü, sürüm |
| `src/dcma/calib/intrinsics.py` | `Intrinsics` veri yapısı; ölçekleme/kırpma altında `K` dönüşümü; FOV'dan türetme |
| `src/dcma/video/manifest.py` | `VideoManifest` veri yapısı; JSON serileştirme |
| `src/dcma/video/normalize.py` | ffprobe sondalama, dönme tespiti, ffmpeg ile kayıpsız kare çıkarma, manifest üretimi |
| `src/dcma/depth/backend.py` | `DepthBackend` soyut arayüzü + disk önbelleği |
| `src/dcma/depth/depth_anything.py` | Depth-Anything-V2 metrik indoor/outdoor backend |
| `src/dcma/vo/features.py` | ORB tespit + eşleme |
| `src/dcma/vo/pose.py` | geri-izdüşüm, PnP RANSAC, `PoseResult` |
| `src/dcma/vo/trajectory.py` | poz birikimi, odometre ve net yer değiştirme |
| `src/dcma/cli.py` | uçtan uca çalıştırma, `trajectory.json` |
| `tests/test_intrinsics.py` | `K` dönüşümleri (saf matematik) |
| `tests/test_manifest.py` | manifest yuvarlak-gidişi |
| `tests/test_normalize.py` | sondalama + oryantasyon (gerçek video gerektirir) |
| `tests/test_orientation_cv2.py` | **cv2 ile ffmpeg oryantasyon uyumu** — Faz 0'ın kilit sorusu |
| `tests/test_depth_backend.py` | derinlik çıktısı şekli/aralığı (GPU gerektirir) |
| `tests/test_features.py` | eşleme sayısı ve filtreleme |
| `tests/test_pose.py` | **sentetik sahne** — poz matematiği, derinlik modelinden bağımsız |
| `tests/test_trajectory.py` | birikim matematiği |
| `tests/conftest.py` | gerçek video / GPU gerektiren testler için fixture ve skip mantığı |

Mevcut `Scripts/Distance_mesurement_4_2.py` → `Scripts/legacy_distance_4_2.py` olarak taşınır.

---

## Task 1: Proje iskeleti ve test altyapısı

**Files:**
- Create: `pyproject.toml`
- Create: `src/dcma/__init__.py`
- Create: `src/dcma/calib/__init__.py`, `src/dcma/video/__init__.py`, `src/dcma/depth/__init__.py`, `src/dcma/vo/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Testi yaz (başarısız olacak)**

`tests/test_package.py`:

```python
def test_package_imports_and_has_version():
    import dcma
    assert isinstance(dcma.__version__, str)
    assert dcma.__version__
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_package.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma'`

- [ ] **Step 3: Minimum kodu yaz**

`pyproject.toml`:

```toml
[project]
name = "dcma"
version = "0.1.0"
description = "Depth-based camera motion analyzer"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "needs_video: gerçek örnek video gerektirir",
    "needs_gpu: CUDA gerektirir",
]
```

`src/dcma/__init__.py`:

```python
__version__ = "0.1.0"
```

Boş `__init__.py` dosyaları:

```bash
mkdir -p src/dcma/{calib,video,depth,vo} tests
touch src/dcma/calib/__init__.py src/dcma/video/__init__.py \
      src/dcma/depth/__init__.py src/dcma/vo/__init__.py
```

`tests/conftest.py`:

```python
import os
import shutil
import pytest

SAMPLE_VIDEO_ENV = "DCMA_SAMPLE_VIDEO"


@pytest.fixture(scope="session")
def sample_video():
    """Gerçek örnek video yolu. DCMA_SAMPLE_VIDEO ile verilir."""
    path = os.environ.get(SAMPLE_VIDEO_ENV)
    if not path or not os.path.isfile(path):
        pytest.skip(f"{SAMPLE_VIDEO_ENV} tanımlı değil veya dosya yok")
    return path


@pytest.fixture(scope="session")
def ffmpeg_available():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe bulunamadı")
    return True


@pytest.fixture(scope="session")
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA yok")
    return 0
```

- [ ] **Step 4: Testin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_package.py -v
```
Beklenen: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/dcma tests/
git commit -m "feat: dcma paket iskeleti ve pytest altyapısı"
```

---

## Task 2: Intrinsics — `K` veri yapısı ve dönüşümleri

Bu saf matematiktir ve metre değerlerinin doğruluğunun temelidir. Ölçekleme veya kırpma
yapıldığında `K` güncellenmezse tüm çıktı sessizce yanlış ölçekte olur.

**Files:**
- Create: `src/dcma/calib/intrinsics.py`
- Test: `tests/test_intrinsics.py`

- [ ] **Step 1: Testi yaz**

`tests/test_intrinsics.py`:

```python
import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics


def base():
    return Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)


def test_matrix_layout():
    K = base().matrix
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, [[1000.0, 0.0, 640.0],
                                   [0.0, 1000.0, 360.0],
                                   [0.0, 0.0, 1.0]])


def test_scaled_half_halves_every_term():
    s = base().scaled(0.5)
    assert (s.fx, s.fy, s.cx, s.cy) == (500.0, 500.0, 320.0, 180.0)
    assert (s.width, s.height) == (640, 360)


def test_scaled_to_max_edge_computes_factor():
    s = base().scaled_to_max_edge(640)
    assert (s.width, s.height) == (640, 360)
    assert s.fx == pytest.approx(500.0)


def test_cropped_shifts_principal_point_only():
    c = base().cropped(x0=100, y0=50, width=1000, height=600)
    assert (c.fx, c.fy) == (1000.0, 1000.0)
    assert (c.cx, c.cy) == (540.0, 310.0)
    assert (c.width, c.height) == (1000, 600)


def test_scale_then_crop_matches_manual_composition():
    k = base().scaled(0.5).cropped(x0=20, y0=10, width=600, height=340)
    assert (k.fx, k.cx, k.cy) == (500.0, 300.0, 170.0)


def test_from_fov_horizontal():
    # 90 derece yatay FOV -> fx = (w/2) / tan(45deg) = w/2
    k = Intrinsics.from_fov(width=1000, height=1000, fov_x_deg=90.0)
    assert k.fx == pytest.approx(500.0)
    assert k.cx == pytest.approx(500.0)
    assert k.cy == pytest.approx(500.0)


def test_rotated_90_swaps_axes_and_maps_principal_point():
    # (u, v) -> (H-1-v, u) saat yonunde 90 derece
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    r = k.rotated(90)
    assert (r.fx, r.fy) == (200.0, 100.0)
    assert (r.cx, r.cy) == (9.0, 10.0)      # (30-1)-20 = 9,  cx = 10
    assert (r.width, r.height) == (30, 40)


def test_rotated_270_is_inverse_of_90():
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    assert k.rotated(90).rotated(270) == k


def test_rotated_180_mirrors_principal_point():
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    r = k.rotated(180)
    assert (r.fx, r.fy) == (100.0, 200.0)
    assert (r.cx, r.cy) == (29.0, 9.0)      # (40-1)-10 = 29, (30-1)-20 = 9
    assert (r.width, r.height) == (40, 30)


def test_rotated_zero_is_identity():
    k = base()
    assert k.rotated(0) == k


def test_rotated_rejects_non_multiples_of_90():
    with pytest.raises(ValueError):
        base().rotated(45)


def test_json_roundtrip():
    k = base()
    assert Intrinsics.from_dict(k.to_dict()) == k


def test_rejects_nonpositive_focal():
    with pytest.raises(ValueError):
        Intrinsics(fx=0.0, fy=1000.0, cx=1.0, cy=1.0, width=10, height=10)


def test_crop_outside_bounds_rejected():
    with pytest.raises(ValueError):
        base().cropped(x0=1000, y0=0, width=1000, height=600)
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_intrinsics.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.calib.intrinsics'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/calib/intrinsics.py`:

```python
"""Kamera iç parametreleri ve görüntü dönüşümleri altında güncellenmesi.

Ölçekleme s icin:  fx' = s*fx, fy' = s*fy, cx' = s*cx, cy' = s*cy
Kırpma (x0, y0) icin:  cx' = cx - x0, cy' = cy - y0   (fx, fy degismez)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"odak uzaklığı pozitif olmalı: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"görüntü boyutu pozitif olmalı: {self.width}x{self.height}")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled(self, s: float) -> "Intrinsics":
        if s <= 0:
            raise ValueError(f"ölçek pozitif olmalı: {s}")
        return Intrinsics(fx=self.fx * s, fy=self.fy * s,
                          cx=self.cx * s, cy=self.cy * s,
                          width=int(round(self.width * s)),
                          height=int(round(self.height * s)))

    def scaled_to_max_edge(self, max_edge: int) -> "Intrinsics":
        longest = max(self.width, self.height)
        if longest <= max_edge:
            return self
        return self.scaled(max_edge / longest)

    def cropped(self, x0: int, y0: int, width: int, height: int) -> "Intrinsics":
        if x0 < 0 or y0 < 0:
            raise ValueError(f"kırpma başlangıcı negatif olamaz: ({x0}, {y0})")
        if x0 + width > self.width or y0 + height > self.height:
            raise ValueError(
                f"kırpma sınır dışı: ({x0}+{width}, {y0}+{height}) > "
                f"({self.width}, {self.height})")
        return Intrinsics(fx=self.fx, fy=self.fy,
                          cx=self.cx - x0, cy=self.cy - y0,
                          width=width, height=height)

    def rotated(self, degrees: int) -> "Intrinsics":
        """90'in katlari kadar dondurulmus goruntunun K'si.

        Piksel merkezi konvansiyonu; saat yonunde donme:
          90  : (u, v) -> (H-1-v, u)
          180 : (u, v) -> (W-1-u, H-1-v)
          270 : (u, v) -> (v, W-1-u)
        """
        d = degrees % 360
        if d == 0:
            return self
        if d == 90:
            return Intrinsics(fx=self.fy, fy=self.fx,
                              cx=(self.height - 1) - self.cy, cy=self.cx,
                              width=self.height, height=self.width)
        if d == 180:
            return Intrinsics(fx=self.fx, fy=self.fy,
                              cx=(self.width - 1) - self.cx,
                              cy=(self.height - 1) - self.cy,
                              width=self.width, height=self.height)
        if d == 270:
            return Intrinsics(fx=self.fy, fy=self.fx,
                              cx=self.cy, cy=(self.width - 1) - self.cx,
                              width=self.height, height=self.width)
        raise ValueError(f"yalnızca 90'ın katları destekleniyor: {degrees}")

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_deg: float) -> "Intrinsics":
        """Yatay görüş açısından kaba K. Kalibrasyon yoksa başlangıç noktası."""
        if not 0.0 < fov_x_deg < 180.0:
            raise ValueError(f"fov_x_deg (0,180) aralığında olmalı: {fov_x_deg}")
        fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(fx=float(d["fx"]), fy=float(d["fy"]),
                   cx=float(d["cx"]), cy=float(d["cy"]),
                   width=int(d["width"]), height=int(d["height"]))
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_intrinsics.py -v
```
Beklenen: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/calib/intrinsics.py tests/test_intrinsics.py
git commit -m "feat: Intrinsics veri yapısı; ölçekleme, kırpma ve döndürme altında K dönüşümü"
```

---

## Task 3: Manifest — normalizasyon kaydı

**Files:**
- Create: `src/dcma/video/manifest.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Testi yaz**

`tests/test_manifest.py`:

```python
import json

import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.video.manifest import VideoManifest


def sample(tmp_path):
    return VideoManifest(
        source_path="/videos/in.mp4",
        source_width=1440,
        source_height=1440,
        source_fps=30.0,
        source_codec="hevc",
        rotation_tag_deg=90,
        rotation_applied=True,
        frames_dir=str(tmp_path / "frames"),
        frame_count=207,
        frame_width=1440,
        frame_height=1440,
        frame_format="png",
        transforms=[{"kind": "autorotate", "degrees": 90}],
        intrinsics=Intrinsics(fx=1100.0, fy=1100.0, cx=720.0, cy=720.0,
                              width=1440, height=1440),
    )


def test_json_roundtrip_preserves_every_field(tmp_path):
    m = sample(tmp_path)
    path = tmp_path / "manifest.json"
    m.save(path)
    assert VideoManifest.load(path) == m


def test_saved_json_is_human_readable(tmp_path):
    m = sample(tmp_path)
    path = tmp_path / "manifest.json"
    m.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["rotation_applied"] is True
    assert d["intrinsics"]["fx"] == 1100.0


def test_intrinsics_may_be_absent(tmp_path):
    m = sample(tmp_path)
    m2 = VideoManifest(**{**m.to_dict(), "intrinsics": None})
    path = tmp_path / "m.json"
    m2.save(path)
    assert VideoManifest.load(path).intrinsics is None


def test_frame_seconds_uses_fps(tmp_path):
    m = sample(tmp_path)
    assert m.frame_seconds(0) == pytest.approx(0.0)
    assert m.frame_seconds(30) == pytest.approx(1.0)


def test_stride_for_interval_rounds_to_at_least_one(tmp_path):
    m = sample(tmp_path)
    assert m.stride_for_interval(0.1) == 3      # 30 fps * 0.1 s
    assert m.stride_for_interval(0.001) == 1    # asla 0 olmaz
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_manifest.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.video.manifest'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/video/manifest.py`:

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_manifest.py -v
```
Beklenen: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/video/manifest.py tests/test_manifest.py
git commit -m "feat: VideoManifest — normalizasyon kaydı ve zaman/adım hesabı"
```

---

## Task 4: Video sondalama ve kayıpsız kare çıkarma

Kareler **PNG** olarak çıkarılır. Mevcut sistem JPG kullanıyordu; JPG kayıplı sıkıştırma
özellik tespitine gürültü katar ve VO doğruluğunu düşürür.

**Files:**
- Create: `src/dcma/video/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Testi yaz**

`tests/test_normalize.py`:

```python
import pytest

from dcma.video.normalize import probe_video, normalize_video


@pytest.mark.needs_video
def test_probe_reports_geometry_and_rotation(sample_video, ffmpeg_available):
    info = probe_video(sample_video)
    assert info.width > 0 and info.height > 0
    assert info.fps > 0
    assert info.codec
    assert info.rotation_deg in (0, 90, 180, 270)


@pytest.mark.needs_video
def test_normalize_writes_png_frames_and_manifest(sample_video, ffmpeg_available, tmp_path):
    m = normalize_video(sample_video, tmp_path, max_frames=8)
    frames = sorted((tmp_path / "frames").glob("*.png"))
    assert len(frames) == 8
    assert m.frame_count == 8
    assert m.frame_format == "png"
    assert m.rotation_applied is True
    assert (tmp_path / "manifest.json").is_file()


@pytest.mark.needs_video
def test_rotation_changes_frame_geometry_when_not_square(sample_video, ffmpeg_available, tmp_path):
    """Dönme uygulandığında kare boyutu kaynağa göre takla atmış olmalı (kare olmayan video)."""
    info = probe_video(sample_video)
    m = normalize_video(sample_video, tmp_path, max_frames=2)
    if info.rotation_deg in (90, 270) and info.width != info.height:
        assert (m.frame_width, m.frame_height) == (info.height, info.width)
    else:
        assert (m.frame_width, m.frame_height) == (info.width, info.height)


@pytest.mark.needs_video
def test_max_edge_downscales_and_updates_intrinsics(sample_video, ffmpeg_available, tmp_path):
    from dcma.calib.intrinsics import Intrinsics
    info = probe_video(sample_video)
    k = Intrinsics.from_fov(info.width, info.height, 70.0)
    m = normalize_video(sample_video, tmp_path, max_frames=2, max_edge=320, intrinsics=k)
    assert max(m.frame_width, m.frame_height) == 320
    assert m.intrinsics is not None
    assert m.intrinsics.width == m.frame_width
    assert m.intrinsics.fx < k.fx
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
export DCMA_SAMPLE_VIDEO=/path/to/sample.mp4
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_normalize.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.video.normalize'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/video/normalize.py`:

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_normalize.py -v
```
Beklenen: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/video/normalize.py tests/test_normalize.py
git commit -m "feat: video normalizasyonu — ffprobe sondalama, autorotate, kayıpsız PNG kareler"
```

---

## Task 5: cv2 oryantasyon uyumu — Faz 0'ın kilit sorusu

Kaynak videolarda `rotate=90` etiketi var ve doğrulandı: `-noautorotate` ile çıkarılan kare
270° döndürülünce varsayılan çıktıyla farkı **tam 0.000**. Bu görev, OpenCV'nin bu etiketi
uygulayıp uygulamadığını **ölçer** ve sonucu kod içinde sabitler. Cevap ne olursa olsun
pipeline'ın tek bir oryantasyonda çalışmasını garanti altına alır.

**Files:**
- Create: `src/dcma/video/orientation.py`
- Test: `tests/test_orientation_cv2.py`

- [ ] **Step 1: Testi yaz**

`tests/test_orientation_cv2.py`:

```python
import numpy as np
import pytest

from dcma.video.normalize import normalize_video, probe_video
from dcma.video.orientation import cv2_applies_rotation, read_frame_upright


@pytest.mark.needs_video
def test_report_whether_cv2_autorotates(sample_video, ffmpeg_available, capsys):
    """Bu test bir davranışı sabitlemez, ÖLÇER ve kayda geçirir."""
    info = probe_video(sample_video)
    applies = cv2_applies_rotation(sample_video)
    with capsys.disabled():
        print(f"\n  rotation_tag={info.rotation_deg}  cv2_autorotate={applies}")
    assert isinstance(applies, bool)


@pytest.mark.needs_video
def test_read_frame_upright_matches_ffmpeg_frame(sample_video, ffmpeg_available, tmp_path):
    """read_frame_upright, ffmpeg'in autorotate çıktısıyla birebir aynı olmalı."""
    import cv2

    m = normalize_video(sample_video, tmp_path, max_frames=5)
    ff = cv2.imread(str(sorted((tmp_path / "frames").glob("*.png"))[3]))
    assert ff is not None

    got = read_frame_upright(sample_video, index=3)
    assert got is not None
    assert got.shape == ff.shape, f"boyut uyuşmazlığı: {got.shape} != {ff.shape}"
    assert np.abs(got.astype(int) - ff.astype(int)).mean() < 1.0
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_orientation_cv2.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.video.orientation'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/video/orientation.py`:

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_orientation_cv2.py -v -s
```
Beklenen: `2 passed`. Çıktıda `rotation_tag=90  cv2_autorotate=<True|False>` satırı görünür —
bu değeri bir sonraki commit mesajına yazın; Faz 0'ın cevabı budur.

- [ ] **Step 5: Commit**

```bash
git add src/dcma/video/orientation.py tests/test_orientation_cv2.py
git commit -m "feat: cv2/ffmpeg oryantasyon uyumu — rotate etiketi ölçülüp kapatıldı"
```

---

## Task 6: Eski script'i legacy olarak taşı

**Files:**
- Modify: `Scripts/Distance_mesurement_4_2.py` → `Scripts/legacy_distance_4_2.py`
- Modify: `README.md`

- [ ] **Step 1: Taşı ve başlığa not ekle**

```bash
git mv Scripts/Distance_mesurement_4_2.py Scripts/legacy_distance_4_2.py
```

`Scripts/legacy_distance_4_2.py` dosyasının en başındaki docstring'i şununla değiştirin:

```python
"""
[ESKİ / LEGACY] 6 Yönlü Hareket Analizi ve Video Annotation Sistemi

Bu script, projenin 4. deneme aşamasıdır ve tarihsel referans olarak korunuyor.
Göreli (affine-invariant) derinlik haritalarının 3×3 bölge ortalamalarını
karşılaştırıp kategorik bir yön etiketi üretir; metre üretmez.

Yerine geçen: src/dcma/ altındaki metrik görsel odometri pipeline'ı.
Gerekçe: docs/superpowers/specs/2026-08-25-metric-6dof-vo-design.md
"""
```

- [ ] **Step 2: README'de yol referansını güncelle**

`README.md` içindeki proje yapısı bloğunda `Distance_mesurement_4_2.py` satırını
`legacy_distance_4_2.py` olarak değiştirin ve yanına `# eski sezgisel yöntem (tarihsel)` yazın.
`Kullanım` bölümündeki `python Scripts/Distance_mesurement_4_2.py` komutunu da güncelleyin.

- [ ] **Step 3: Testlerin hâlâ geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/ -v
```
Beklenen: tüm testler geçer (bu görev kod yolu değiştirmedi)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: eski sezgisel script'i legacy olarak işaretle"
```

---

## Task 7: Derinlik backend arayüzü ve Depth-Anything metrik

Doğrulanmış davranış: `pipeline("depth-estimation")` çıktısı
`{'predicted_depth': Tensor, 'depth': PIL.Image}`. `predicted_depth` girdi boyutunda gelir.
İki ortamda ayrı ayrı ölçüldü ve **bit-birebir aynı** sonucu verdi — transformers 5.5.0
(CPU) ve `dcma` ortamındaki 5.15.1 (GPU): `Indoor-Large` gerçek iç mekân karesinde
min 1.451 / medyan 2.638 / max 5.910 m. Buna rağmen kod boyutu **varsaymaz, doğrular**
(`ensure_size`).

Ölçülen hız (RTX 4090, 1440×1440): `Indoor-Large` 65 ms/kare, `Indoor-Base` 38 ms/kare.
207 karelik bir video Large ile ~14 saniyede işlenir; bu yüzden varsayılan `large`.

**Files:**
- Create: `src/dcma/depth/backend.py`
- Create: `src/dcma/depth/depth_anything.py`
- Test: `tests/test_depth_backend.py`

- [ ] **Step 1: Testi yaz**

`tests/test_depth_backend.py`:

```python
import numpy as np
import pytest
from PIL import Image

from dcma.depth.depth_anything import DepthAnythingMetric, CHECKPOINTS


def test_checkpoint_table_covers_both_scenes_and_sizes():
    for scene in ("indoor", "outdoor"):
        for size in ("base", "large"):
            assert (scene, size) in CHECKPOINTS


def test_unknown_scene_rejected():
    with pytest.raises(ValueError):
        DepthAnythingMetric(scene="underwater", size="base", lazy=True)


def test_auto_scene_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        DepthAnythingMetric(scene="auto", size="base", lazy=True)


@pytest.mark.needs_gpu
def test_predict_returns_metric_metres(cuda_device):
    backend = DepthAnythingMetric(scene="indoor", size="base", device=cuda_device)
    img = Image.new("RGB", (518, 518), (110, 120, 130))
    d = backend.predict(img)
    assert d.dtype == np.float32
    assert d.shape == (518, 518), f"girdi boyutunda olmalı, geldi: {d.shape}"
    assert np.isfinite(d).all()
    assert d.min() > 0.0
    assert d.max() < 200.0


@pytest.mark.needs_gpu
def test_cache_returns_identical_array_without_recompute(cuda_device, tmp_path):
    backend = DepthAnythingMetric(scene="indoor", size="base",
                                  device=cuda_device, cache_dir=tmp_path)
    img = Image.new("RGB", (256, 256), (60, 90, 120))
    first = backend.predict_cached(img, key="f0")
    assert (tmp_path / "f0.npy").is_file()
    second = backend.predict_cached(img, key="f0")
    np.testing.assert_array_equal(first, second)
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_depth_backend.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.depth.depth_anything'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/depth/backend.py`:

```python
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
        np.save(path, depth.astype(np.float16))
        return depth


def ensure_size(depth: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Derinlik haritasını (genişlik, yükseklik) boyutuna getirir."""
    width, height = size_wh
    if depth.shape == (height, width):
        return depth.astype(np.float32)
    import cv2
    return cv2.resize(depth.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR)
```

`src/dcma/depth/depth_anything.py`:

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_depth_backend.py -v
```
Beklenen: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/depth tests/test_depth_backend.py
git commit -m "feat: metrik derinlik backend'i — Depth-Anything V2 indoor/outdoor + önbellek"
```

---

## Task 8: Özellik tespiti ve eşleme

**Files:**
- Create: `src/dcma/vo/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Testi yaz**

`tests/test_features.py`:

```python
import numpy as np

from dcma.vo.features import detect_and_match


def _textured_image(seed=0, size=480):
    """ORB'nin guvenilir kose bulabilecegi sentetik desen.

    Saf rastgele gurultu ORB icin kotu bir girdidir (belirgin kose yok);
    cizilmis dikdortgen ve daireler tekrarlanabilir kose uretir.
    """
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(20, size - 60, 2)
        w, h = rng.integers(15, 45, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      int(rng.integers(120, 255)), -1)
    for _ in range(40):
        x, y = rng.integers(20, size - 20, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 14)),
                   int(rng.integers(80, 255)), -1)
    return img


def test_shifted_image_matches_recover_the_shift():
    base = _textured_image()
    dx, dy = 7, 4
    shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)

    p1, p2 = detect_and_match(base, shifted, n_features=2000)
    assert len(p1) == len(p2)
    assert len(p1) > 50

    delta = p2 - p1
    # cogunluk gercek kaymayi bulmali; medyan saglam bir ozet
    assert abs(np.median(delta[:, 0]) - dx) < 1.5
    assert abs(np.median(delta[:, 1]) - dy) < 1.5


def test_returns_float32_pixel_coordinates():
    a, b = _textured_image(1), _textured_image(1)
    p1, p2 = detect_and_match(a, b, n_features=500)
    assert p1.dtype == np.float32 and p2.dtype == np.float32
    assert p1.ndim == 2 and p1.shape[1] == 2


def test_blank_images_yield_no_matches():
    blank = np.zeros((256, 256), dtype=np.uint8)
    p1, p2 = detect_and_match(blank, blank, n_features=500)
    assert len(p1) == 0 and len(p2) == 0
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_features.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.vo.features'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/vo/features.py`:

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_features.py -v
```
Beklenen: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/vo/features.py tests/test_features.py
git commit -m "feat: ORB tabanlı özellik tespiti ve eşleme"
```

---

## Task 9: Poz kestirimi — sentetik testle doğrulanmış

**Bu planın en kritik görevi.** Sentetik sahne, bilinen `K` ve bilinen `(R, t)` ile
üretilir; derinlik modeli hiç devreye girmez. Bu sayede bir sayı yanlış olduğunda
suçlunun algoritma mı yoksa derinlik modeli mi olduğu belli olur.

**Files:**
- Create: `src/dcma/vo/pose.py`
- Test: `tests/test_pose.py`

- [ ] **Step 1: Testi yaz**

`tests/test_pose.py`:

```python
import cv2
import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics
from dcma.vo.pose import backproject, estimate_pose


def _scene(R, t, n=400, seed=0):
    """Bilinen (R, t) ile sentetik sahne: 3B noktalar ve iki karedeki izdüşümleri."""
    rng = np.random.default_rng(seed)
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0,
                   width=1000, height=1000)
    X1 = np.stack([rng.uniform(-2.0, 2.0, n),
                   rng.uniform(-2.0, 2.0, n),
                   rng.uniform(2.0, 6.0, n)], axis=1)
    X2 = (R @ X1.T).T + t

    def proj(X):
        return np.stack([K.fx * X[:, 0] / X[:, 2] + K.cx,
                         K.fy * X[:, 1] / X[:, 2] + K.cy], axis=1).astype(np.float32)

    return K, X1, proj(X1), proj(X2)


def test_pure_forward_translation_recovers_exact_metres():
    d = 0.25
    R = np.eye(3)
    t = np.array([0.0, 0.0, -d])       # kamera +z'de d ilerlerse noktalar -d kayar
    K, X1, _, p2 = _scene(R, t)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.inliers > 350
    np.testing.assert_allclose(res.R, R, atol=1e-6)
    np.testing.assert_allclose(res.t, t, atol=1e-6)
    np.testing.assert_allclose(res.C, [0.0, 0.0, d], atol=1e-6)
    assert res.forward == pytest.approx(d, abs=1e-6)
    assert res.right == pytest.approx(0.0, abs=1e-6)
    assert res.up == pytest.approx(0.0, abs=1e-6)


def test_pure_right_translation_reports_positive_right():
    d = 0.4
    R = np.eye(3)
    t = np.array([-d, 0.0, 0.0])       # kamera +x'te (saga) d giderse noktalar -d kayar
    K, X1, _, p2 = _scene(R, t, seed=3)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.right == pytest.approx(d, abs=1e-6)
    assert res.forward == pytest.approx(0.0, abs=1e-6)


def test_pure_up_translation_reports_positive_up():
    d = 0.3
    R = np.eye(3)
    t = np.array([0.0, d, 0.0])        # y ASAGI oldugu icin yukari hareket +y kaydirir
    K, X1, _, p2 = _scene(R, t, seed=4)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    assert res.up == pytest.approx(d, abs=1e-6)


def test_rotation_and_translation_recovered_together():
    R = cv2.Rodrigues(np.array([0.01, np.deg2rad(4.0), -0.02]))[0]
    t = np.array([0.05, -0.02, -0.30])
    K, X1, _, p2 = _scene(R, t, seed=7)

    res = estimate_pose(X1, p2, K)
    assert res is not None
    np.testing.assert_allclose(res.R, R, atol=1e-6)
    np.testing.assert_allclose(res.t, t, atol=1e-6)
    assert res.reproj_err < 1e-3


def test_backproject_inverts_projection():
    K = Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)
    X = np.array([[0.5, -0.25, 3.0], [-1.0, 0.75, 5.0]])
    uv = np.stack([K.fx * X[:, 0] / X[:, 2] + K.cx,
                   K.fy * X[:, 1] / X[:, 2] + K.cy], axis=1).astype(np.float32)
    depth = np.zeros((480, 640), dtype=np.float32)
    for (u, v), z in zip(np.round(uv).astype(int), X[:, 2]):
        depth[v, u] = z

    P3d, z = backproject(depth, K, uv)
    np.testing.assert_allclose(z, X[:, 2], atol=1e-4)
    np.testing.assert_allclose(P3d, X, atol=1e-2)


def test_too_few_points_returns_none():
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    res = estimate_pose(np.zeros((3, 3)), np.zeros((3, 2), dtype=np.float32), K)
    assert res is None
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_pose.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.vo.pose'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/vo/pose.py`:

```python
"""Kare çiftleri arası poz kestirimi.

Konvansiyonlar (OpenCV kamera çerçevesi):
  +x saga, +y ASAGI, +z ileri
  solvePnP cikti: X_cam2 = R @ X_cam1 + t
  kamera-2 merkezinin kare-1 koordinatindaki yeri: C = -R.T @ t
  rapor eksenleri: ileri = C[2], saga = C[0], yukari = -C[1]
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dcma.calib.intrinsics import Intrinsics

MIN_POINTS = 6


@dataclass
class PoseResult:
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    inliers: int
    reproj_err: float

    @property
    def forward(self) -> float:
        return float(self.C[2])

    @property
    def right(self) -> float:
        return float(self.C[0])

    @property
    def up(self) -> float:
        return float(-self.C[1])

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.C))


def backproject(depth: np.ndarray, K: Intrinsics,
                pts2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2B noktaları derinlik ve K ile kamera koordinatlarında 3B'ye taşır."""
    u = np.asarray(pts2d, dtype=np.float64)[:, 0]
    v = np.asarray(pts2d, dtype=np.float64)[:, 1]
    h, w = depth.shape[:2]
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    z = depth[vi, ui].astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=1), z


def estimate_pose(P3d: np.ndarray, p2: np.ndarray, K: Intrinsics,
                  reproj_thresh: float = 2.0, iterations: int = 2000,
                  confidence: float = 0.999) -> PoseResult | None:
    """kare-1'in 3B noktaları ile kare-2'deki izdüşümlerinden pozu çözer."""
    obj = np.ascontiguousarray(P3d, dtype=np.float64)
    img = np.ascontiguousarray(p2, dtype=np.float64)
    if len(obj) < MIN_POINTS or len(obj) != len(img):
        return None

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K.matrix, None,
        reprojectionError=reproj_thresh,
        iterationsCount=iterations,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < MIN_POINTS:
        return None

    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K.matrix, None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).ravel()

    proj, _ = cv2.projectPoints(obj[idx], rvec, tvec, K.matrix, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1).mean())

    return PoseResult(R=R, t=t, C=(-R.T @ t), inliers=int(len(idx)), reproj_err=err)
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_pose.py -v
```
Beklenen: `6 passed`. Herhangi biri başarısızsa **devam etmeyin** — işaret veya
konvansiyon hatası vardır ve sonraki her sayı yanlış olur.

- [ ] **Step 5: Commit**

```bash
git add src/dcma/vo/pose.py tests/test_pose.py
git commit -m "feat: PnP RANSAC poz kestirimi, sentetik sahne testleriyle doğrulanmış"
```

---

## Task 10: Yörünge birikimi — odometre ve net yer değiştirme

İki ayrı büyüklük raporlanır ve **karıştırılmamaları önemlidir**:

- **Odometre:** her adımın kendi kamera çerçevesindeki ileri/sağa/yukarı bileşenlerinin
  toplamı. "Toplam 5 m ileri gitti" sorusunun cevabı budur.
- **Net yer değiştirme:** başlangıç çerçevesinde ölçülen baştan-sona vektör.
  "Şu an başlangıçtan ne kadar uzakta" sorusunun cevabı. Kamera bir tur atıp
  başladığı yere dönerse odometre büyür ama net yer değiştirme sıfıra yaklaşır.

**Files:**
- Create: `src/dcma/vo/trajectory.py`
- Test: `tests/test_trajectory.py`

- [ ] **Step 1: Testi yaz**

`tests/test_trajectory.py`:

```python
import cv2
import numpy as np
import pytest

from dcma.vo.trajectory import Trajectory


def test_starts_at_origin_with_identity_pose():
    tr = Trajectory()
    np.testing.assert_allclose(tr.positions[-1], [0.0, 0.0, 0.0])
    assert tr.step_count == 0


def test_two_forward_steps_accumulate():
    tr = Trajectory()
    for _ in range(2):
        tr.add_step(np.eye(3), np.array([0.0, 0.0, -0.1]))

    np.testing.assert_allclose(tr.positions[-1], [0.0, 0.0, 0.2], atol=1e-9)
    odo = tr.odometer
    assert odo["forward"] == pytest.approx(0.2)
    assert odo["right"] == pytest.approx(0.0)
    assert odo["up"] == pytest.approx(0.0)
    np.testing.assert_allclose(tr.net_displacement, [0.0, 0.0, 0.2], atol=1e-9)


def test_accumulation_equals_manual_matrix_product():
    """Birikim tam olarak T_{k+1} = T_k @ inv(dT_k) olmalı."""
    rng = np.random.default_rng(11)
    steps = []
    for _ in range(4):
        rvec = rng.normal(scale=0.05, size=3)
        R = cv2.Rodrigues(rvec)[0]
        t = rng.normal(scale=0.2, size=3)
        steps.append((R, t))

    tr = Trajectory()
    for R, t in steps:
        tr.add_step(R, t)

    expected = np.eye(4)
    for R, t in steps:
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        expected = expected @ np.linalg.inv(dT)

    np.testing.assert_allclose(tr.poses[-1], expected, atol=1e-10)


def test_loop_returns_near_origin_but_odometer_grows():
    """Ileri git, 180 don, ayni kadar ileri git -> baslangica yakin don."""
    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -1.0]))
    R180 = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0]
    tr.add_step(R180, np.array([0.0, 0.0, 0.0]))
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -1.0]))

    assert np.linalg.norm(tr.net_displacement) < 1e-6
    assert tr.odometer["forward"] == pytest.approx(2.0, abs=1e-9)
    assert tr.path_length == pytest.approx(2.0, abs=1e-9)


def test_export_dict_has_per_step_records():
    tr = Trajectory()
    tr.add_step(np.eye(3), np.array([0.0, 0.0, -0.1]), inliers=120, reproj_err=0.7)
    d = tr.to_dict()
    assert d["step_count"] == 1
    assert d["steps"][0]["inliers"] == 120
    assert d["steps"][0]["forward"] == pytest.approx(0.1)
    assert "odometer" in d and "net_displacement" in d
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_trajectory.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.vo.trajectory'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/vo/trajectory.py`:

```python
"""Poz birikimi ve metre cinsinden özetler.

Poz konvansiyonu: T_wc[k] = dunya-kameradan donusum, dunya = ilk keyframe.
Adim (R, t) su anlama gelir: X_cam(k+1) = R @ X_cam(k) + t
Dolayisiyla T_wc[k+1] = T_wc[k] @ inv([R|t]).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Trajectory:
    def __init__(self) -> None:
        self.poses: list[np.ndarray] = [np.eye(4)]
        self._steps: list[dict[str, Any]] = []

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def positions(self) -> np.ndarray:
        return np.array([T[:3, 3] for T in self.poses])

    @property
    def net_displacement(self) -> np.ndarray:
        return self.positions[-1] - self.positions[0]

    @property
    def path_length(self) -> float:
        return float(sum(s["distance"] for s in self._steps))

    @property
    def odometer(self) -> dict[str, float]:
        return {
            "forward": float(sum(s["forward"] for s in self._steps)),
            "right": float(sum(s["right"] for s in self._steps)),
            "up": float(sum(s["up"] for s in self._steps)),
        }

    def add_step(self, R: np.ndarray, t: np.ndarray,
                 inliers: int | None = None,
                 reproj_err: float | None = None,
                 frame_from: int | None = None,
                 frame_to: int | None = None) -> None:
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()

        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        self.poses.append(self.poses[-1] @ np.linalg.inv(dT))

        C = -R.T @ t
        self._steps.append({
            "frame_from": frame_from,
            "frame_to": frame_to,
            "forward": float(C[2]),
            "right": float(C[0]),
            "up": float(-C[1]),
            "distance": float(np.linalg.norm(C)),
            "inliers": inliers,
            "reproj_err": reproj_err,
        })

    def to_dict(self) -> dict[str, Any]:
        net = self.net_displacement
        return {
            "step_count": self.step_count,
            "odometer": self.odometer,
            "net_displacement": {
                "forward": float(net[2]),
                "right": float(net[0]),
                "up": float(-net[1]),
                "magnitude": float(np.linalg.norm(net)),
            },
            "path_length": self.path_length,
            "positions": self.positions.tolist(),
            "steps": self._steps,
        }
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_trajectory.py -v
```
Beklenen: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dcma/vo/trajectory.py tests/test_trajectory.py
git commit -m "feat: yörünge birikimi — odometre ve net yer değiştirme ayrımıyla"
```

---

## Task 11: CLI — uçtan uca çalıştırma

**Files:**
- Create: `src/dcma/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Testi yaz**

`tests/test_cli.py`:

```python
import json

import pytest

from dcma.cli import build_parser, run


def test_parser_requires_scene():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "x.mp4", "--out", "o"])


def test_parser_rejects_unknown_scene():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "x.mp4", "--out", "o", "--scene", "space"])


def test_parser_defaults():
    args = build_parser().parse_args(
        ["--video", "x.mp4", "--out", "o", "--scene", "indoor"])
    assert args.size == "large"
    assert args.fov_x == 70.0
    assert args.interval == 0.15


@pytest.mark.needs_video
@pytest.mark.needs_gpu
def test_end_to_end_writes_trajectory(sample_video, ffmpeg_available, cuda_device, tmp_path):
    args = build_parser().parse_args([
        "--video", sample_video, "--out", str(tmp_path),
        "--scene", "indoor", "--size", "base",
        "--max-frames", "24", "--max-edge", "512",
    ])
    run(args)

    path = tmp_path / "trajectory.json"
    assert path.is_file()
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["step_count"] > 0
    assert "odometer" in d
    assert all(k in d["odometer"] for k in ("forward", "right", "up"))
```

- [ ] **Step 2: Başarısız olduğunu doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_cli.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'dcma.cli'`

- [ ] **Step 3: Uygulamayı yaz**

`src/dcma/cli.py`:

```python
"""Uçtan uca çalıştırma: video -> normalize -> derinlik -> poz -> trajectory.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dcma.calib.intrinsics import Intrinsics
from dcma.depth.depth_anything import DepthAnythingMetric
from dcma.video.normalize import normalize_video, probe_video
from dcma.vo.features import detect_and_match
from dcma.vo.pose import backproject, estimate_pose
from dcma.vo.trajectory import Trajectory

# Gecerli derinlik araliklari sahneye gore degisir: ic mekan modeli ~20 m'ye,
# dis mekan modeli ~80 m'ye kadar egitildi. Araligin disi guvenilmez.
DEPTH_RANGE_M = {
    "indoor": (0.3, 15.0),
    "outdoor": (1.0, 60.0),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcma", description="Derinlik tabanlı metrik kamera hareket analizi")
    p.add_argument("--video", required=True, help="girdi videosu (herhangi format)")
    p.add_argument("--out", required=True, help="çıktı dizini")
    p.add_argument("--scene", required=True, choices=["indoor", "outdoor", "auto"],
                   help="sahne türü; 'auto' Faz 5'te gelecek")
    p.add_argument("--size", default="large", choices=["base", "large"])
    p.add_argument("--fov-x", dest="fov_x", type=float, default=70.0,
                   help="kalibrasyon yoksa yatay görüş açısı (derece)")
    p.add_argument("--interval", type=float, default=0.15,
                   help="kare çiftleri arası hedef süre (saniye)")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p.add_argument("--max-edge", dest="max_edge", type=int, default=None)
    p.add_argument("--device", type=int, default=0)
    return p


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(args.video)
    k_source = Intrinsics.from_fov(info.width, info.height, args.fov_x)
    manifest = normalize_video(args.video, out_dir,
                               max_edge=args.max_edge,
                               max_frames=args.max_frames,
                               intrinsics=k_source)
    K = manifest.intrinsics
    if K is None:
        raise RuntimeError("manifest intrinsics içermiyor")

    frames = sorted(Path(manifest.frames_dir).glob("*.png"))
    stride = manifest.stride_for_interval(args.interval)
    print(f"kare sayısı={len(frames)}  adım={stride} "
          f"({args.interval}s @ {manifest.source_fps:.2f} fps)")

    backend = DepthAnythingMetric(scene=args.scene, size=args.size,
                                  device=args.device,
                                  cache_dir=out_dir / "depth_cache")
    min_depth, max_depth = DEPTH_RANGE_M[args.scene]
    traj = Trajectory()
    skipped: list[dict] = []

    for i in range(0, len(frames) - stride, stride):
        j = i + stride
        img_i = Image.open(frames[i]).convert("RGB")
        gray_i = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        gray_j = cv2.imread(str(frames[j]), cv2.IMREAD_GRAYSCALE)

        p1, p2 = detect_and_match(gray_i, gray_j)
        if len(p1) < 20:
            skipped.append({"from": i, "to": j, "reason": "yetersiz eşleşme",
                            "matches": int(len(p1))})
            continue

        depth_i = backend.predict_cached(img_i, key=f"{i:06d}")
        P3d, z = backproject(depth_i, K, p1)
        valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
        if valid.sum() < 20:
            skipped.append({"from": i, "to": j, "reason": "yetersiz geçerli derinlik",
                            "valid": int(valid.sum())})
            continue

        res = estimate_pose(P3d[valid], p2[valid], K)
        if res is None:
            skipped.append({"from": i, "to": j, "reason": "PnP çözülemedi"})
            continue

        traj.add_step(res.R, res.t, inliers=res.inliers,
                      reproj_err=res.reproj_err, frame_from=i, frame_to=j)

    payload = traj.to_dict()
    payload["skipped"] = skipped
    payload["manifest"] = manifest.to_dict()
    (out_dir / "trajectory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    odo = payload["odometer"]
    net = payload["net_displacement"]
    print(f"\nadım={payload['step_count']}  atlanan={len(skipped)}")
    print(f"odometre : ileri={odo['forward']:+.3f} m  sağa={odo['right']:+.3f} m  "
          f"yukarı={odo['up']:+.3f} m")
    print(f"net      : ileri={net['forward']:+.3f} m  sağa={net['right']:+.3f} m  "
          f"yukarı={net['up']:+.3f} m  (|{net['magnitude']:.3f}| m)")
    print(f"yol uzunluğu: {payload['path_length']:.3f} m")
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/test_cli.py -v
```
Beklenen: `4 passed`

- [ ] **Step 5: Tüm test paketini çalıştır**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m pytest tests/ -v
```
Beklenen: tüm testler geçer, hiçbiri hata vermez

- [ ] **Step 6: Commit**

```bash
git add src/dcma/cli.py tests/test_cli.py
git commit -m "feat: uçtan uca CLI — video'dan trajectory.json'a"
```

---

## Task 12: Gerçek videoda çalıştır ve sonucu değerlendir

Bu görev kod yazmaz; pipeline'ın gerçek veride ne ürettiğini **kayda geçirir.**

**Files:**
- Create: `docs/results/2026-08-25-first-run.md`

- [ ] **Step 1: İç mekân videosunda çalıştır**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY -m dcma.cli \
  --video "/path/to/sample.mp4" \
  --out /tmp/dcma_run1 \
  --scene indoor --size large --interval 0.15 --max-edge 768
```

- [ ] **Step 2: Çıktıyı incele**

```bash
env -u VIRTUAL_ENV -u PYTHONPATH $DPY - <<'PY'
import json
d = json.load(open("/tmp/dcma_run1/trajectory.json"))
print("adim       :", d["step_count"], " atlanan:", len(d["skipped"]))
print("odometre   :", {k: round(v, 3) for k, v in d["odometer"].items()})
print("net        :", {k: round(v, 3) for k, v in d["net_displacement"].items()})
print("yol        :", round(d["path_length"], 3), "m")
inl = [s["inliers"] for s in d["steps"] if s["inliers"]]
err = [s["reproj_err"] for s in d["steps"] if s["reproj_err"]]
if inl:
    print("inlier     : ort=%.0f min=%d max=%d" % (sum(inl)/len(inl), min(inl), max(inl)))
if err:
    print("reproj px  : ort=%.2f max=%.2f" % (sum(err)/len(err), max(err)))
from collections import Counter
print("atlama sebepleri:", Counter(s["reason"] for s in d["skipped"]))
PY
```

- [ ] **Step 3: Sonucu belgele**

`docs/results/2026-08-25-first-run.md` dosyasına şunları yazın:
kullanılan komut, adım/atlanan sayıları, odometre ve net değerler, ortalama inlier ve
reprojection hatası, atlama sebeplerinin dağılımı. Sayıların **makul görünüp görünmediğine
dair yorumunuzu** da ekleyin — videoda kamera kabaca ne kadar hareket ediyordu?

**Bu bir doğrulama değildir.** Değerlerin gerçekten doğru olduğu ancak Faz 3'te
(TUM RGB-D + şeritmetreyle ölçülmüş çekim) belirlenir. Bu adım yalnızca pipeline'ın
uçtan uca çalıştığını ve çıktının büyüklük mertebesinin saçma olmadığını kaydeder.

- [ ] **Step 4: Commit**

```bash
git add docs/results/2026-08-25-first-run.md
git commit -m "docs: ilk uçtan uca çalıştırma sonuçları"
```

---

## Faz 1 çıkış koşulu

- `tests/` tamamen geçiyor; `tests/test_pose.py` sentetik testleri 1e-6 toleransla geçiyor
- `dcma.cli` gerçek iç mekân videosunda `trajectory.json` üretiyor
- `rotate=90` sorusu ölçülüp kapatılmış, cevap commit mesajında kayıtlı
- Atlanan kare sayısı ve sebepleri çıktıda görünüyor

**Faz 2'ye geçmeden önce:** ilk çalıştırma sonuçları gözden geçirilip, paralaks tabanlı
keyframe seçimi ve sağlamlık kapılarının hangi eşiklerle kurulacağına gerçek veriye
bakarak karar verilecek.
