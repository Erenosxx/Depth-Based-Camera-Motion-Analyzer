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
