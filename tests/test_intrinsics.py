import numpy as np
import pytest

from dcma.calib.intrinsics import Intrinsics


def base():
    return Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)


def assert_fields_approx(got: Intrinsics, want: Intrinsics) -> None:
    """Alan alan approx: kayan nokta kalıntısı '==' ile nesne karşılaştırmasını bozar."""
    assert (got.fx, got.fy, got.cx, got.cy) == pytest.approx(
        (want.fx, want.fy, want.cx, want.cy))
    assert (got.width, got.height) == (want.width, want.height)


def test_matrix_layout():
    K = base().matrix
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, [[1000.0, 0.0, 640.0],
                                   [0.0, 1000.0, 360.0],
                                   [0.0, 0.0, 1.0]])


def test_scaled_half_scales_focal_and_shifts_principal_point():
    # Piksel merkezi i+0.5: cx' = s*(cx+0.5)-0.5 = 0.5*640.5-0.5 = 319.75
    # (cy' = 0.5*360.5-0.5 = 179.75). Ana nokta odak gibi salt yarıya inmez.
    s = base().scaled(0.5)
    assert (s.fx, s.fy) == (500.0, 500.0)
    assert (s.cx, s.cy) == pytest.approx((319.75, 179.75))
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
    # scaled(0.5) -> cx=319.75, cy=179.75; kırpma ana noktayı öteler:
    # 319.75-20 = 299.75,  179.75-10 = 169.75
    k = base().scaled(0.5).cropped(x0=20, y0=10, width=600, height=340)
    assert k.fx == pytest.approx(500.0)
    assert (k.cx, k.cy) == pytest.approx((299.75, 169.75))


def test_from_fov_horizontal():
    # 90 derece yatay FOV -> fx = (w/2) / tan(45deg) = w/2
    # Ana nokta dönmeye göre değişmeyen merkez: (w-1)/2 = 499.5 (w/2 değil;
    # rotated(180) cx -> (w-1)-cx dönüşümünün tek sabit noktası bu).
    k = Intrinsics.from_fov(width=1000, height=1000, fov_x_deg=90.0)
    assert k.fx == pytest.approx(500.0)
    assert k.cx == pytest.approx(499.5)
    assert k.cy == pytest.approx(499.5)


def test_rotated_90_swaps_axes_and_maps_principal_point():
    # (u, v) -> (H-1-v, u) saat yonunde 90 derece
    k = Intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0, width=40, height=30)
    r = k.rotated(90)
    assert (r.fx, r.fy) == (200.0, 100.0)
    assert (r.cx, r.cy) == (9.0, 10.0)      # (30-1)-20 = 9,  cx = 10
    assert (r.width, r.height) == (30, 40)


def test_rotated_270_is_inverse_of_90():
    # Ana nokta bilerek ikili tabanda tam gösterilemeyen bir değer: tam '=='
    # karşılaştırması ~1e-15 kalıntıyla patlar, bu yüzden approx kullanıyoruz.
    k = Intrinsics(fx=100.0, fy=200.0, cx=1 / 3, cy=0.1, width=40, height=30)
    r = k.rotated(90).rotated(270)
    assert_fields_approx(r, k)


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


# --- regresyon: iki farklı piksel konvansiyonunun karışmasını yakalayan testler ---


@pytest.mark.parametrize("scale", [0.5, 0.25])
def test_centred_principal_point_stays_centred_under_scale(scale):
    """Tam ortalanmış K her yeniden boyutlandırmadan sonra ortalanmış kalmalı."""
    w, h = 1280, 720
    k = Intrinsics(fx=1000.0, fy=1000.0,
                   cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, width=w, height=h)
    s = k.scaled(scale)
    assert s.cx == pytest.approx((s.width - 1) / 2.0)
    assert s.cy == pytest.approx((s.height - 1) / 2.0)


def test_from_fov_is_invariant_under_180_rotation():
    """cx=(W-1)/2 olmasının nedeni: rotated(180)'in tek sabit noktası bu."""
    k = Intrinsics.from_fov(width=1000, height=1000, fov_x_deg=90.0)
    assert_fields_approx(k.rotated(180), k)


def test_scaled_to_max_edge_uses_per_axis_factors():
    """1918x1080 -> 640x360: yuvarlamalar farklı olduğu için fx ve fy ayrı ölçek kullanır."""
    k = Intrinsics(fx=1000.0, fy=1000.0,
                   cx=(1918 - 1) / 2.0, cy=(1080 - 1) / 2.0,
                   width=1918, height=1080)
    s = k.scaled_to_max_edge(640)
    assert (s.width, s.height) == (640, 360)
    assert s.fx == pytest.approx(1000.0 * 640 / 1918)
    assert s.fy == pytest.approx(1000.0 * 360 / 1080)
    assert s.fx != pytest.approx(s.fy)


def test_rotated_90_four_times_is_identity():
    k = Intrinsics(fx=100.0, fy=200.0, cx=1 / 3, cy=0.1, width=40, height=30)
    r = k.rotated(90).rotated(90).rotated(90).rotated(90)
    assert_fields_approx(r, k)


def test_rotation_angle_is_normalised_modulo_360():
    k = Intrinsics(fx=100.0, fy=200.0, cx=1 / 3, cy=0.1, width=40, height=30)
    assert k.rotated(-90) == k.rotated(270)
    assert k.rotated(450) == k.rotated(90)


@pytest.mark.parametrize("kwargs", [
    {"fx": float("nan")},
    {"fx": float("inf")},
    {"cx": float("nan")},
])
def test_rejects_non_finite_values(kwargs):
    """nan/inf sessizce geçerse fx=inf her noktayı 0.0 metre yapar."""
    fields = dict(fx=1000.0, fy=1000.0, cx=10.0, cy=10.0, width=100, height=100)
    fields.update(kwargs)
    with pytest.raises(ValueError):
        Intrinsics(**fields)


@pytest.mark.parametrize("wh", [(0, 100), (100, 0), (-10, 100)])
def test_resized_to_rejects_nonpositive_dimensions(wh):
    with pytest.raises(ValueError):
        base().resized_to(*wh)


def test_from_fov_long_edge_is_orientation_independent():
    """Kamera spesifikasyonu sensörün uzun eksenine aittir; portre ve yatay
    kaynak aynı odak uzaklığını vermelidir."""
    land = Intrinsics.from_fov_long_edge(3840, 2176, 70.0)
    port = Intrinsics.from_fov_long_edge(2176, 3840, 70.0)
    assert land.fx == pytest.approx(port.fx)
    assert land.fy == pytest.approx(port.fy)


def test_from_fov_long_edge_on_portrait_gives_plausible_fovs():
    """Natif portre kaynakta 70° uzun (dikey) eksene gitmeli, yatay daralmalı.

    Regresyon: from_fov(2176, 3840, 70) açıyı KISA eksene uyguluyordu ve dikey
    FOV 102° çıkıyordu — hiçbir normal kamera için mümkün değil. Bu hata
    ofis_gezisi videosunda dönmeyi 3840/2176 = 1.765 kat şişirmişti (gerçek
    90° dönüş -159° olarak raporlanıyordu).
    """
    import math
    k = Intrinsics.from_fov_long_edge(2176, 3840, 70.0)
    hfov = 2 * math.degrees(math.atan(k.width / 2 / k.fx))
    vfov = 2 * math.degrees(math.atan(k.height / 2 / k.fy))
    assert vfov == pytest.approx(70.0, abs=1e-9)
    assert hfov == pytest.approx(43.28, abs=0.05)
    assert hfov < 90.0 and vfov < 90.0


def test_from_fov_long_edge_equals_from_fov_on_landscape():
    """Genişlik uzun eksen olduğunda iki fonksiyon aynı K'yı vermeli."""
    assert (Intrinsics.from_fov_long_edge(3840, 2160, 70.0)
            == Intrinsics.from_fov(3840, 2160, 70.0))


def test_from_fov_long_edge_rejects_bad_angle():
    with pytest.raises(ValueError):
        Intrinsics.from_fov_long_edge(100, 200, 0.0)
    with pytest.raises(ValueError):
        Intrinsics.from_fov_long_edge(100, 200, 180.0)
