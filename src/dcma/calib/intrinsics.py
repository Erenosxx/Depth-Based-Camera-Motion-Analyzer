"""Kamera iç parametreleri ve görüntü dönüşümleri altında güncellenmesi.

Konvansiyon: piksel merkezi tam sayı indekste, yani `i` indeksli pikselin
merkezi sürekli koordinatta `i + 0.5`, görüntü sürekli olarak [0, W] aralığını
kaplar ve dönmeye göre değişmeyen görüntü merkezi `(W-1)/2` indeksindedir.
Geri projeksiyon bu konvansiyonu varsayar:
    z = depth[v, u];  x = (u - cx) * z / fx;  y = (v - cy) * z / fy

Yeniden boyutlandırma sürekli alanı [0, W] -> [0, sW] örnekler, dolayısıyla
sürekli koordinat s ile ölçeklenir:
    u_yeni + 0.5 = s * (u_eski + 0.5)   =>   cx' = s*(cx + 0.5) - 0.5
Odak uzaklıkları doğrudan ölçeklenir:  fx' = s*fx, fy' = s*fy
Kırpma (x0, y0) icin:  cx' = cx - x0, cy' = cy - y0   (fx, fy degismez)
"""

from __future__ import annotations

import math
import numbers
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
        for name, value in (("fx", self.fx), ("fy", self.fy),
                            ("cx", self.cx), ("cy", self.cy)):
            if not math.isfinite(value):
                raise ValueError(
                    f"iç parametreler sonlu olmalı: {name}={value} "
                    f"(nan/inf sessizce hatalı metrik üretir)")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"odak uzaklığı pozitif olmalı: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"görüntü boyutu pozitif olmalı: {self.width}x{self.height}")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def resized_to(self, width: int, height: int) -> "Intrinsics":
        """Verilen boyuta yeniden ölçeklenmiş görüntünün K'si.

        Eksen başına ölçek hedef boyuttan türetilir; boyutlar bağımsız
        yuvarlandığında fx ve fy'nin gerçek örnekleme oranıyla tutarlı kalmasını
        sağlar. Piksel merkezi i+0.5 konvansiyonu:
            u_yeni = sx * (u_eski + 0.5) - 0.5
        """
        if not isinstance(width, numbers.Integral) or not isinstance(height, numbers.Integral):
            raise ValueError(f"hedef boyut tam sayı olmalı: {width!r}x{height!r}")
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"hedef boyut pozitif olmalı: {width}x{height}")
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(fx=self.fx * sx, fy=self.fy * sy,
                          cx=sx * (self.cx + 0.5) - 0.5,
                          cy=sy * (self.cy + 0.5) - 0.5,
                          width=width, height=height)

    def scaled(self, s: float) -> "Intrinsics":
        """s katsayısıyla yeniden boyutlandırılmış görüntünün K'si.

        Hedef boyutlar yuvarlanır ve gerçek ölçek bu yuvarlanmış boyutlardan
        türetilir; böylece K.width/height her zaman kare boyutuyla eşleşir.
        """
        if s <= 0:
            raise ValueError(f"ölçek pozitif olmalı: {s}")
        return self.resized_to(int(round(self.width * s)),
                               int(round(self.height * s)))

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
        """Yatay görüş açısından kaba K. Kalibrasyon yoksa başlangıç noktası.

        Sensör sürekli olarak W genişliği kaplar, yani yarım açıklık W/2 piksel:
            fx = (W/2) / tan(fov_x/2)
        Ana nokta piksel merkezi konvansiyonunda dönmeye göre değişmeyen
        merkezdir: cx = (W-1)/2, cy = (H-1)/2.

        DİKKAT: fy = fx kare piksel (SAR = 1) varsayar; anamorfik ya da birim
        olmayan SAR'lı kaynaklar için geçerli DEĞİLDİR. Bu fonksiyon bildirilen
        her metreyi ölçekleyen yedek yoldur; gerçek kalibrasyon tercih edilmeli.
        """
        if not 0.0 < fov_x_deg < 180.0:
            raise ValueError(f"fov_x_deg (0,180) aralığında olmalı: {fov_x_deg}")
        fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
        return cls(fx=fx, fy=fx,
                   cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                   width=width, height=height)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(fx=float(d["fx"]), fy=float(d["fy"]),
                   cx=float(d["cx"]), cy=float(d["cy"]),
                   width=int(d["width"]), height=int(d["height"]))
