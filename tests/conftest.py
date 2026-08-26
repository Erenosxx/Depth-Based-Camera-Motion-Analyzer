import os
import shutil
from pathlib import Path

import pytest

SAMPLE_VIDEO_ENV = "DCMA_SAMPLE_VIDEO"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_local_env() -> None:
    """local.env (git'e girmez) içindeki değişkenleri, yoksa ortamda setdefault ile yükle."""
    path = _REPO_ROOT / "local.env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


_load_local_env()


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
