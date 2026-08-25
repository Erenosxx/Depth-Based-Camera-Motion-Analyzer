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
