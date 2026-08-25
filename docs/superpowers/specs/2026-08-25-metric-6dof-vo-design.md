# Metrik 6-DoF Kamera Hareket Analizi — Tasarım

**Tarih:** 2026-08-25
**Durum:** Onaylandı
**Kapsam:** İç mekân videosundan metre cinsinden 6-DoF kamera hareketi; ölçülebilir doğruluk; format-bağımsız girdi.

---

## 1. Problem

Mevcut sistem ([`Scripts/Distance_mesurement_4_2.py`](../../../Scripts/Distance_mesurement_4_2.py)) her kare için
göreli derinlik haritası üretip 3×3 bölge ortalamalarının farkına bakıyor ve 7 kategoriden birini
(`İLERİ`, `GERİ`, `SAĞA`, `SOLA`, `YUKARI`, `AŞAĞI`, `BELİRSİZ`) etiket olarak veriyor.

Bu yaklaşımdan metre çıkmaz. Üç bağımsız sebep:

1. **Monoküler ölçek belirsizliği.** Tek kameradan mutlak ölçek geometrik olarak kurtarılamaz.
   Ölçek dışarıdan bir kaynaktan gelmek zorunda.
2. **Kullanılan checkpoint affine-invariant.** `Depth-Anything-V2-base-hf` için gerçek derinlik
   `Dₜ = aₜ·D̂ₜ + bₜ` biçimindedir ve `aₜ, bₜ` kareden kareye değişir. Dolayısıyla
   `D̂[i+5] − D̂[i]` farkı gerçek hareketin yanına `(aᵢ₊₅ − aᵢ)` ve `(bᵢ₊₅ − bᵢ)` gürültüsünü katar.
   **Gözlenen %65 `BELİRSİZ` oranının kök nedeni budur**; katı `all()` kuralı yalnızca semptomdur.
3. **Bölge ortalaması farkı geometrik bir büyüklük değil.** "Ortalama derinlik 0.03 azaldı" ile
   "kamera 12 cm ilerledi" arasında sahne-bağımsız bir dönüşüm yoktur.

Ek olarak mevcut kodda üç somut kusur var:

- Ekrana yazılan yön sayaçları **kümülatif değil**; `movement_counts` analiz bitince dolduğu için
  her karede aynı nihai toplamlar basılıyor.
- Döngü her adımda `kare[i]` ve `kare[i+5]` için ayrı çıkarım yapıyor; neredeyse her kare
  **iki kez** modele giriyor.
- Kaynak videolarda `TAG:rotate=90` var. ffmpeg bu etiketi uygular, OpenCV `VideoCapture`
  yok sayar. Kareler ffmpeg ile çıkarılıp analiz edildiği, annotation ise cv2 ile yapıldığı için
  analiz edilen görüntü ile üzerine yazılan görüntü 90° farklı olabilir — bu doğruysa
  `SAĞA/SOLA` ile `YUKARI/AŞAĞI` sistematik olarak yer değiştirmiş durumdadır.

## 2. Hedef ve kapsam dışı

**Hedef**

- Metre cinsinden 6-DoF hareket: 3 öteleme (ileri/geri, sağ/sol, yukarı/aşağı) + 3 dönme.
- Kümülatif yer değiştirme ve yörünge.
- Girdi videosu herhangi bir format / codec / çözünürlük / oran / fps olabilir.
- Üretilen metre değerlerinin doğruluğunun ölçülmüş olması.
- İç mekân öncelikli; dış mekân modeli hazır, geçiş anahtarı mimaride mevcut.

**Kapsam dışı (bu spec'te değil)**

- Mutlak coğrafi konum (enlem/boylam), GPS ikamesi.
- Araç / uzun mesafe senaryosu.
- Döngü kapatma, bundle adjustment, tam SLAM.
- Gerçek zamanlı çalışma.

## 3. Mimari kararı

**Seçilen: RGB-D PnP Görsel Odometri + essential-matrix çapraz kontrolü.**

Ana yol: metrik derinlikle 3B nokta bulutu kur, `cv2.solvePnPRansac` ile poz çöz, birik.
Doğrulama yolu: `cv2.findEssentialMat` + `recoverPose` ile dönmeyi bağımsız olarak da hesapla.
İki dönme kestirimi eşikten fazla ayrışırsa o kare çifti şüpheli işaretlenip atlanır.

Değerlendirilen alternatifler:

| Alternatif | Neden seçilmedi |
|---|---|
| Geometri öncüllü hibrit (dönme+yön essential'dan, ölçek derinlikten) | Saf dönmede ve düzlemsel sahnede dejenere; proje "derinlik tabanlı" kimliğinden uzaklaşır |
| Olgun SLAM'i sarmalama (ORB-SLAM3 / MASt3R-SLAM) | C++ derleme yükü; ölçüm altyapısı kurulmadan optimize etmek olur; Faz 5 malzemesi |
| Yalın PnP (çapraz kontrol yok) | Derinlik modelinin bozulduğu kareler sessizce poza karışır |

**Neden PnP (3B-2B), 3B-3B hizalama değil:** 3B-3B'de iki karenin derinlik gürültüsü de hataya
girer. PnP'de yalnızca kaynak karenin derinliği kullanılır, hedef kare saf 2B kalır.

## 4. Modül yapısı

```
src/dcma/
├── __init__.py
├── cli.py                    # argparse giriş noktası
├── video/
│   ├── normalize.py          # her format → kanonik kareler + manifest
│   └── manifest.py           # manifest şeması (dataclass + JSON serileştirme)
├── calib/
│   ├── intrinsics.py         # K sağlayıcı; resize/crop sonrası K güncelleme
│   └── calibrate.py          # satranç tahtası kalibrasyon aracı
├── depth/
│   ├── backend.py            # soyut arayüz: predict(image) -> metre haritası
│   ├── depth_anything.py     # indoor/outdoor metrik checkpoint'ler
│   └── depth_pro.py          # metrik derinlik + odak uzaklığı tahmini
├── scene/
│   └── classify.py           # indoor|outdoor kancası (Faz 5'te dolar)
├── vo/
│   ├── features.py           # ORB eşleme (Faz 5'te LightGlue)
│   ├── keyframe.py           # paralaks tabanlı kare seçimi
│   ├── pose.py               # PnP RANSAC + essential çapraz kontrol + kapılar
│   └── trajectory.py         # poz birikimi, metre ayrıştırma, dışa aktarma
├── viz/
│   ├── annotate.py           # videoya kümülatif metre + mini yörünge
│   └── plot.py               # matplotlib yörünge grafiği
└── eval/
    ├── tum.py                # TUM RGB-D yükleyici
    └── metrics.py            # ATE / RPE
```

Mevcut script `Scripts/legacy_distance_4_2.py` adına taşınır ve korunur; README'deki
"projenin evrimi" anlatısının kanıtıdır.

**Sınırlar:** her modül tek bir işten sorumludur ve ötekilerin içine bakmadan kullanılabilir.
`depth/backend.py` soyut arayüzü sayesinde Depth-Anything ile DepthPro birbirinin yerine geçebilir;
`calib/intrinsics.py` K'nın nereden geldiğini VO'dan gizler.

## 5. Veri akışı

```
video (herhangi format)
   ├─▶ normalize          → frames/ + manifest.json
   ├─▶ intrinsics çözümü  → K
   ├─▶ metrik derinlik    → Dₜ [metre]   (önbellekli)
   ├─▶ keyframe seçimi    (paralaks bandı)
   ├─▶ eşleme → PnP RANSAC → ΔT + çapraz kontrol + kapılar
   └─▶ birikim            → trajectory.json · report.md · plot.png · annotated.mp4
```

## 6. Belirleyici tasarım kararları

### 6.1 Manifest intrinsics'i taşır

Resize veya crop `K`'yı değiştirir: ölçekleme `s` için `fx' = s·fx`, `fy' = s·fy`,
`cx' = s·cx`, `cy' = s·cy`; kırpma `(x₀, y₀)` için `cx' = cx − x₀`, `cy' = cy − y₀`.

Manifest uygulanan her dönüşümü kaydeder; `intrinsics.py` `K`'yı buna göre günceller.
Bu zorunlu, çünkü eldeki videolar iki farklı çekim modunda (1440×1440 1:1 ve 3840×2160 16:9)
ve **bu iki modun `fx` ve `cx` değerleri farklıdır.**

### 6.2 Adım zamanda sabit, karede değil

Mevcut kod sabit 5 kare kullanıyor: 30 fps'te 167 ms, 60 fps'te 83 ms. Stride zamandan
hesaplanır, böylece aynı kod farklı fps'li videolarda aynı davranır.

### 6.3 Paralaks tabanlı keyframe seçimi

Sabit adım iki yönden hatalı: yavaş harekette baseline dejenere olacak kadar küçük,
hızlı harekette eşleme kopar. Aday kareler ileri doğru taranır ve eşleşen noktaların
**medyan piksel kayması** hedef banda düşen ilk kare seçilir. VO doğruluğuna en çok
etki eden karar budur.

Varsayılanlar (yapılandırılabilir, 1440 px kısa kenar referansıyla):

| Parametre | Değer |
|---|---|
| Hedef medyan paralaks bandı | 8–40 px |
| İleri tarama üst sınırı | 30 kare |
| Bant üstünde kalınırsa | bir önceki adaya dön (paralaks çok büyük, eşleme kopar) |
| Bant altında kalınırsa | tarama sınırına kadar ilerle, hâlâ altındaysa kareyi atla (hareket yok) |

Bant, kısa kenarla ölçeklenir: `‖kayma‖ / kısa_kenar` oranı sabit tutulur, böylece
farklı çözünürlüklerde aynı davranış elde edilir.

### 6.4 Derinlik önbelleği

Her kare modele bir kez girer; sonuç fp16 olarak önbelleğe alınır. Mevcut koddaki
iki kat fazla hesap ortadan kalkar.

### 6.5 Ölçek yanlılığı düzeltmesi

Metrik derinlik modelleri sistematik yanlılık taşır. TUM RGB-D gerçek derinlik ground
truth'u içerdiğinden modelin yanlılığı ölçülüp tek bir çarpan olarak çıkarılır ve
kendi videolara da uygulanır.

### 6.6 Sağlamlık kapıları — sessiz başarısızlık yok

Her kare çifti için sırayla:

| Kapı | Varsayılan eşik |
|---|---|
| RANSAC inlier sayısı | ≥ 30 |
| Ortalama reprojection hatası | ≤ 2.0 px |
| PnP dönmesi ile essential dönmesi arasındaki açı | ≤ 5° |
| Fiziksel makullük: `‖t‖ / Δt` | ≤ 3.0 m/s (el kamerası) |
| Geçerli derinlik aralığı (iç mekân) | 0.3 m < z < 15.0 m |

Eşikler yapılandırılabilir. Kapıya takılan kareler `report.md`'de **sebebiyle birlikte**
listelenir. Mevcut `BELİRSİZ` etiketinin sorunu "bilmiyorum" ile "hata yaptım"ı aynı
kutuya atmasıydı; bu ayrım korunur.

### 6.7 indoor/outdoor kancası

`--scene {indoor,outdoor,auto}`. `indoor` ve `outdoor` ilgili metrik checkpoint'i seçer.
`auto` tanımlıdır ancak Faz 5'e kadar `NotImplementedError` fırlatır. Kanca şimdi konur ki
özellik eklenirken mimari değişmesin.

### 6.8 Konvansiyonlar

- OpenCV kamera çerçevesi: `+x` sağa, `+y` **aşağı**, `+z` ileri.
- Rapor eksenleri: `ileri = +z`, `sağa = +x`, `yukarı = −y`.
- Poz birikimi: `T_wc` (dünya-kameradan), 4×4; dünya çerçevesi ilk keyframe'dir.
- `solvePnPRansac` çıktısı `X_cam2 = R·X_cam1 + t` verir; kamera merkezinin kare-1
  koordinatındaki yeri `C = −Rᵀt`.

### 6.9 Normalizasyon politikası

- Dönme metadata'sı **açıkça uygulanır** (ffmpeg autorotate), sonuç manifest'e yazılır.
- Konum/GPS metadata'sı silinir.
- Varsayılan olarak yerel çözünürlük korunur; `--max-edge` verilirse ölçeklenir ve `K` güncellenir.
- Derinlik haritası olarak `predicted_depth` kullanılır. Pipeline'ın `depth` anahtarı
  (normalize edilmiş PIL görüntüsü) **kullanılmaz** — metrik değildir.
- transformers 5.5.0'da `pipeline("depth-estimation")` çıktısını girdi boyutuna kendisi
  getiriyor (doğrulandı: 1440×1440 girdi → `predicted_depth` shape `(1440, 1440)`).
  Kod bunu **varsaymaz, doğrular**; boyut uyuşmazsa bilinear resize uygular.

## 7. Çıktılar

| Dosya | İçerik |
|---|---|
| `trajectory.json` | kare başına poz (R, t), kümülatif metre, güven bayrakları, kapı sonuçları |
| `report.md` | toplam ileri/geri/sağ/sol/yukarı/aşağı metre; atlanan kare sayısı ve sebepleri; ortalama inlier |
| `plot.png` | üstten görünüm yörünge + yükseklik profili |
| `annotated.mp4` | canlı **gerçekten kümülatif** metre + mini yörünge overlay |

## 8. Test stratejisi

- **Sentetik sahne (kritik).** Bilinen `K` ve bilinen `ΔT` ile üretilmiş nokta bulutu ve
  mükemmel derinlik. Kestirilen poz analitik doğruya eşit çıkmalıdır. Bu test algoritma
  hatasını derinlik modeli hatasından **ayırır**; bir sayı yanlışsa suçlunun hangisi
  olduğu bilinir.
- **Birim.** `K` dönüşümleri (resize/crop), manifest yuvarlak-gidişi, poz birikim matematiği.
- **Entegrasyon.** TUM RGB-D iç mekân dizisi; ATE ve RPE hesaplanır ve raporlanır.
- **Gerçek.** Şeritmetreyle ölçülmüş kontrollü kendi çekimi.

### Kabul ölçütleri

| Test | Ölçüt |
|---|---|
| Sentetik poz kestirimi | öteleme hatası < 1e-6 m, dönme hatası < 1e-6 ° |
| `K` dönüşüm birim testleri | tamamı geçer |
| TUM RGB-D | ATE ve RPE **ölçülür ve raporlanır**; Faz 3 bu değerleri temel çizgi olarak sabitler |
| Kontrollü kendi çekim (2 m öteleme) | hedef: %10 içinde |

TUM RGB-D için baştan sayısal bir eşik taahhüt edilmiyor; ilk ölçüm temel çizgiyi kurar,
sonraki iyileştirmeler bu çizgiye göre değerlendirilir.

## 9. Fazlar

| Faz | İçerik | Çıkış koşulu |
|---|---|---|
| **0** | opencv kurulumu; `normalize.py` + manifest; `rotate=90` hipotezinin doğrulanması | Kanonik kareler üretiliyor; dönme sorusu kesin yanıtlanmış |
| **1** | derinlik backend + intrinsics + PnP VO + `trajectory.json` | Kendi videosunda metre değerleri üretiliyor; sentetik test geçiyor |
| **2** | keyframe seçimi, sağlamlık kapıları, çapraz kontrol, `report.md` + `plot.png` | Atlanan kareler sebebiyle raporlanıyor |
| **3** | TUM RGB-D değerlendirmesi + ölçek yanlılığı düzeltmesi | ATE/RPE ölçülmüş ve kayda geçmiş |
| **4** | `annotate.py` yenilenmesi, README ve demo güncellemesi | Demo GIF'li public README |
| **5** | indoor/outdoor `auto`, LightGlue, dış mekân | Açık uçlu; her madde kendi spec'ini alır |

### Bu spec'in uygulama planı kapsamı

İlk uygulama planı **yalnızca Faz 0 ve Faz 1'i** kapsar; bu iki faz birlikte
"kendi videosunda metre üreten, sentetik testle doğrulanmış bir pipeline" anlamına gelir
ve tek planda yürütülebilir büyüklüktedir. Faz 2 ve sonrası, Faz 1 tamamlanıp gerçek
çıktı görüldükten sonra kendi planlarını alır — ölçülmemiş bir sistemin ilerisini
planlamak varsayım üstüne varsayım koymak olur.

## 10. Bağımlılıklar

Mevcut ortam: RTX 4090 24 GB, torch 2.6.0+cu124, transformers 5.5.0, numpy 2.4.4,
scipy 1.17.1, matplotlib 3.10.9, huggingface_hub 1.14.0.

**Eklenecek:** `opencv-python` (ortamda yok — mevcut script bu ortamda hiç çalışmıyor),
`evo` (yörünge değerlendirmesi), `pyyaml` (yapılandırma).

**İndirilmiş modeller:** `Depth-Anything-V2-Metric-{Indoor,Outdoor}-{Base,Large}-hf`,
`apple/DepthPro-hf`. Varsayılan iç mekân checkpoint'i `Indoor-Large`.

### Doğrulanmış ortam bulguları (2026-08-25)

**API teyit edildi.** `pipeline("depth-estimation")` transformers 5.5.0'da çalışıyor.
Çıktı anahtarları `['predicted_depth', 'depth']`; `predicted_depth` bir `torch.Tensor` ve
girdi boyutunda. `Indoor-Large` checkpoint'i gerçek iç mekân karesinde
min 1.451 / medyan 2.638 / max 5.910 üretti — **metre ölçeğinde ve makul.**

**`rotate=90` hipotezi doğrulandı.** Aynı kare ffmpeg'in `-noautorotate` ile çıkarılıp
270° döndürüldüğünde varsayılan (autorotate) çıktısına **birebir eşit** (ortalama mutlak
fark 0.000). Yani videoda gerçek bir 90° dönme var ve iki okuma yolu tam 90° sapıyor.
**OpenCV sorusu Faz 0'da ölçüldü ve kapandı.** Bu makinede (OpenCV 5.0)
`cv2.VideoCapture` dönme etiketini **uyguluyor** (`CAP_PROP_ORIENTATION_AUTO`).
Bağımsız doğrulama: çıplak cv2 karesi ile ffmpeg autorotate karesinin ortalama mutlak
farkı döndürmesiz **0.000**, 90°/180°/270° döndürmelerde 55.7 / 54.1 / 55.7.

Sonuç: eski pipeline'da şüphelenilen analiz-annotation oryantasyon uyumsuzluğu
**fiilen mevcut değildi**; `SAĞA/SOLA` ile `YUKARI/AŞAĞI` yer değiştirmemiş. Ancak bu
garanti OpenCV derlemesine bağlı, dile ait değil — `src/dcma/video/orientation.py`
cevabı sabitlemez, her çağrıda ölçer ve yalnızca gerekiyorsa döndürür.

**Ayrı ortam zorunluluğu.** `LLM_training` ortamında `nvidia-cudnn-cu13` (9.19.0.56)
dosyalarını `nvidia/cudnn/lib/` altına yazarak `nvidia-cudnn-cu12` (9.1.0.70) üzerine
yazmış. torch 2.6.0+cu124 bu yoldan yüklediği için cuDNN 9.19 alıyor; sürücü 535.261.03
ile initialize olmuyor ve **her `conv2d` çağrısı `CUDNN_STATUS_NOT_INITIALIZED` ile
patlıyor** (`matmul` etkilenmiyor). Derinlik modelleri konvolüsyon ağı olduğundan GPU'da
çalışmıyor. `nvidia-cudnn-cu13` paketinin `Required-by` alanı boş; muhtemel kaynak
`torchao 0.15.0` (torch 2.6.0+cu124 ile uyumsuz olduğu uyarısı veriyor).

**Karar:** `LLM_training` ortamına dokunulmaz (aktif Gemma eğitimi barındırıyor).
Proje kendi `dcma` conda ortamında geliştirilir; HF cache paylaşımlı kaldığı için
indirilmiş 5.1 GB model yeniden inmez.

## 11. Intrinsics stratejisi

`K` değiştirilebilir bir bileşendir; iki kaynak desteklenir ve aynı videoda kıyaslanabilir:

1. **Kalibrasyon JSON'u** — `calib/calibrate.py` ile satranç tahtasından üretilir.
   Her çekim modu (1:1, 16:9) için ayrı kalibrasyon gerekir.
2. **DepthPro tahmini** — odak uzaklığını görüntüden tahmin eder; kalibrasyon yoksa kullanılır.

Öncelik: kalibrasyon varsa o, yoksa tahmin. Geliştirme tahminle başlar; kalibrasyon
sonradan eklenip doğruluk artışı ölçülür.
