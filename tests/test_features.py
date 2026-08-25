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
