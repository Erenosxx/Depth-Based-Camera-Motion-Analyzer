<div align="center">

# 🎥 Depth-Based Camera Motion Analyzer

**Tek bir monoküler videodan, IMU/GPS olmadan, kamera hareketini metre cinsinden 6-DoF olarak çıkarır.**

[Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf)
metrik derinlik + ORB eşleme + PnP RANSAC ile kareler arası poz çözülür; yörünge birikir
ve sonuç annotate videoya, grafiklere yazılır.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
![Status](https://img.shields.io/badge/status-V1%20araştırma%20prototipi-orange)

</div>

---

## 🎬 Demo

İç mekân, 3840×2160 @ 60 fps, ~35 sn’lik bir çekimin annotate özeti (3× hız, 5 fps, 59 kare) ve yörünge grafiği:

<div align="center">

![Annotate video: renkli izler, HUD ve minimap, tüm çekim](assets/demo.gif)

</div>

**`demo.gif` — tüm çekim, seyrek karelerle.** Ne görünüyor:

| Bölge | Anlamı |
|:---|:---|
| **Renkli noktalar + kuyruk** | Her nokta ayrı bir özellik; rengi sabittir. Arkada kalan çizgi o noktanın **ekrandaki gerçek yoludur** (kare-kare Lucas-Kanade), iki uç arasında çekilmiş düz kiriş değildir. |
| **Sol üst yazılar** | Kare numarası ve o ana kadar biriken metre: `ileri` / `saga` / `yukari`. |
| **Sol alt kare (minimap)** | Kuş bakışı yörünge. **Yeşil** = çekimin başlangıcı, **kırmızı** = kameranın şu anki yeri, beyaz çizgi = o ana kadar gidilen yol. Kamera yeşil–kırmızı mesafesine göre kayar; mesafe azalınca (geri dönüş) görüş donar. Noktalar kenara yakın durur, değmez. |
| **Sağ alt etiket** | O anki baskın öteleme yönü: İLERİ, GERİ, SAĞA, SOLA, YUKARI, AŞAĞI veya DURAGAN. |

<div align="center">
<img src="assets/plot.png" width="720" alt="Üstten yörünge ve yükseklik profili">
</div>

**`plot.png` — tüm çekimin özeti (tek bakışta).** Ne görünüyor:

| Panel | Anlamı |
|:---|:---|
| **Üstten görünüm** (sol) | Dünya düzlemi: yatay eksen **sağa (m)**, dikey eksen **ileri (m)**. Yeşil nokta başlangıç, kırmızı nokta bitiş. Çizgi kameranın yerde bıraktığı izdir. |
| **Yükseklik profili** (sağ) | Adım adım **yukarı (m)**. El kamerasının salınımı ve kat çıkışı burada okunur. |
| **Başlık** | `net` = başlangıç–bitiş kuş uçuşu mesafe. `yol` = odometre (gidilen toplam yol). Gidip dönünce yol büyür, net küçük kalır — bu beklenen ayrım. |

Asıl teslim dosyası `Result/<ad>.mp4` (H.264, kaynak FPS). `Result/<ad>/frames/` ham kare önbelleğidir; çıktı değildir.

---

## 📌 Ne Yapar?

Klasik optik akış 2B piksel kaymasına bakar ve sahne hareketi ile kamera hareketini karıştırır.
Bu sistem **metrik derinlikle 3B nokta** kurar, sonraki karedeki 2B izdüşümden `solvePnPRansac`
ile pozu çözer ve metre cinsinden biriktirir.

Çıktı kategorik etiket değil: **ileri / sağa / yukarı metre**, kümülatif yörünge ve anlık yön.

> Kamera ileri giderse derinlik azalır ve PnP `+z` ötelemesi üretir. Sağa kayınca noktalar
> sola kayar; bu, bölge ortalaması sezgisinden bağımsız geometrik bir ölçüdür.

### Anlık yön (videodaki etiket)

| Yön | Ekrandaki renk | Anlamı |
|:---|:---|:---|
| **İLERİ** | 🟢 Yeşil | Kamera sahneye yaklaşıyor (`+z`) |
| **GERİ** | 🔴 Kırmızı | Kamera sahneden uzaklaşıyor |
| **SAĞA** | 🔵 Mavi | Kamera sağa öteleniyor (`+x`) |
| **SOLA** | 🩵 Cyan | Kamera sola öteleniyor |
| **YUKARI** | 🟣 Magenta | Kamera yukarı öteleniyor (`−y`, OpenCV) |
| **AŞAĞI** | 🟡 Sarı | Kamera aşağı öteleniyor |
| **DURAGAN** | ⚪ Gri | Bu adımda baskın öteleme yok |

Bunlar 3 öteleme ekseni × 2 yön. Dönme (pan / tilt / roll) pozun içinde çözülür ama
etiket olarak ayrıştırılmaz — bkz. [Bilinen Sınırlamalar](#-bilinen-sınırlamalar).

---

## ⚙️ Nasıl Çalışır?

```
video (her format)
   → normalize     PNG kare + manifest (dönme uygulanır, K güncellenir)
   → metrik derinlik   Depth-Anything V2 Indoor/Outdoor (önbellekli, metre)
   → ORB eşleme → geri izdüşüm → PnP RANSAC → ΔT
   → yörünge birikimi
   → trajectory.json + plot.png + annotated.mp4 (H.264)
```

1. **Normalizasyon.** ffprobe + ffmpeg; `rotate` etiketi açıkça uygulanır. Kareler kayıpsız PNG.
2. **Metrik derinlik.** Varsayılan `Indoor-Large`. Her kare modele bir kez girer; sonuç disk önbelleğinde.
3. **Poz.** Kaynak karenin 3B noktaları + hedef karenin 2B eşleşmeleri → `solvePnPRansac`.
4. **Birikim.** Odometre (adımların toplamı) ile net yer değiştirme (başlangıç–bitiş) ayrı tutulur.
5. **Görselleştirme.** Yön HUD’u yörüngeden; nokta izleri ise kare-kare Lucas-Kanade’den (VO’dan bağımsız).

---

## 🚀 Kurulum

```bash
git clone https://github.com/Erenosxx/Depth-Based-Camera-Motion-Analyzer.git
cd Depth-Based-Camera-Motion-Analyzer

# önerilen: ayrı conda ortamı (dcma). LLM_training ortamına paket kurmayın.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 💡 `torch`'u kendi CUDA sürümünüze göre kurun:
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

Metrik Indoor/Outdoor checkpoint’leri ilk çalışmada Hugging Face’den iner.
`ffmpeg` / `ffprobe` annotate video (H.264) için gerekli.

Bu makinede (`dev-host`) çalıştırma: `VIRTUAL_ENV` tuzaklarına karşı mutlak yol kullanın
(aşağıdaki Kullanım bloğu).

---

## 🧪 Kullanım

Ham videolar `Data/`, işlenmiş çıktılar `Result/` altına gider (ikisi de git’te yok).

```bash
export DPY=/path/to/envs/dcma/bin/python

env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 PYTHONPATH=src $DPY -m dcma.cli \
  --video "Data/girdi.mp4" \
  --out "Result/calisma_adi" \
  --scene indoor \
  --size large \
  --interval 0.15 \
  --max-edge 768
```

`--scene indoor|outdoor` zorunlu (`auto` henüz yok). Konsolda odometre, net yer değiştirme
ve atlanan kare sayısı basılır.

Yalnızca görseli yenilemek (derinlik tekrar çalışmaz):

```bash
env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 PYTHONPATH=src $DPY -m dcma.viz.annotate \
  --run Result/calisma_adi
```

Eski 3×3 sezgisel yöntem tarihsel referans olarak duruyor: `Scripts/legacy_distance_4_2.py`.
Metre üretmez.

---

## 📤 Çıktı

| Dosya | İçerik |
|:---|:---|
| `Result/<ad>.mp4` | Asıl teslim: H.264, kaynak FPS, iz + HUD + minimap |
| `Result/<ad>/annotated.mp4` | Aynı video, koşu klasöründe |
| `Result/<ad>/trajectory.json` | Adım adım R, t, metre, atlanan kareler |
| `Result/<ad>/plot.png` | Üstten yörünge + yükseklik (yukarıdaki gibi) |
| `Result/<ad>/preview.png` | Ortadaki kareden still (yerel; README’de GIF kullanılır) |
| `Result/<ad>/frames/` | Ara bellek PNG — çıktı değil |
| `Result/<ad>/depth_cache/` | Derinlik `.npy` önbelleği |

**Referans koşu (V1):** 3840×2160 @ 60 fps, ~35 sn, 2112 kare, `--max-edge 768`.
234 adım, 0 atlama. Örnek: odometre ileri +1.64 m, yol 28.98 m, net 1.91 m
(ileri/geri/sol/sağ çekimi — yol > net beklenir). Şeritmetre / TUM ile henüz doğrulanmadı.

---

## 🔬 Teknik Detaylar

| Bileşen | Değer |
|:---|:---|
| Derinlik | `Depth-Anything-V2-Metric-Indoor-Large-hf` (varsayılan) |
| Poz | ORB + `solvePnPRansac` + refine |
| Görselleştirme izi | Shi-Tomasi + Lucas-Kanade, noktaya özel renk |
| Video yazımı | ffmpeg `libx264` `yuv420p` (OpenCV `mp4v` FPS’i bozuyordu) |
| Minimap | Yeşil–kırmızı mesafe; geri dönüşte görüş donar; kenar payı %8 |
| Kamera ekseni | OpenCV: `+x` sağ, `+y` aşağı, `+z` ileri |

---

## ⚠️ Bilinen Sınırlamalar

Bu bir **araştırma prototipidir** (V1).

| # | Sınırlama | Neden / Etki |
|:--|:---|:---|
| 1 | **Ölçek modelden gelir** | Monoküler geometri mutlak metre veremez; Indoor checkpoint yanlı olabilir. TUM / şeritmetre henüz yok. |
| 2 | **`K` kaba** | Kalibrasyon yoksa yatay FOV 70° varsayılır. Yanlış `K` tüm metreyi çarpar. |
| 3 | **Dönme etiketi yok** | Poz 6-DoF çözülür; HUD yalnızca öteleme yönünü yazar. |
| 4 | **Keyframe yok** | Sabit zaman aralığı (`--interval`). Çok yavaş/hızlı harekette eşleme zayıflar. |
| 5 | **Sağlamlık kapıları eksik** | Essential-matrix çapraz kontrolü ve hız kapısı Faz 2. |
| 6 | **Çözünürlük** | `--max-edge` küçültmesi görseli ve `K`’yı değiştirir. |

---

## 🗺️ Yol Haritası

- [x] Metrik derinlik + PnP VO + `trajectory.json`
- [x] CLI, Data/Result ayrımı, H.264 annotate, minimap
- [ ] TUM RGB-D ATE/RPE + ölçek yanlılığı
- [ ] Paralaks keyframe + sağlamlık kapıları
- [ ] Dönme (pan / tilt / roll) HUD
- [ ] Kontrollü şeritmetre çekimi (%10 hedef)

---

## 🧭 Projenin Evrimi

| Deneme | Yaklaşım | Sonuç |
|:--|:---|:---|
| **1** | Lucas-Kanade + grid (Gradio) | Kamera vs nesne ayrımı yok |
| **2** | 6 yön + zoom, kümülatif eşikler | Sahneye aşırı duyarlı |
| **3** | JPG kare dışa aktarma | Yöntem değişmedi |
| **4** | Göreli Depth Anything + 3×3 bölge farkı | Yön etiketi var, metre yok, ~%65 BELİRSİZ |
| **5 (V1)** | **Metrik derinlik + PnP görsel odometri** | Metre + yörünge + annotate video |

Kilit içgörü aynı: **2B kaymayı sınıf etiketine çevirmek yerine 3B yapıyı ölçmek.**
V1 bunu geometrik poza bağlar.

---

## 📂 Proje Yapısı

```
Depth-Based-Camera-Motion-Analyzer/
├── Data/                     # ham videolar (repoya girmez)
├── Result/                   # işlenmiş çıktılar (repoya girmez)
├── Scripts/
│   ├── legacy_distance_4_2.py
│   └── make_demo_gif.sh
├── src/dcma/                 # metrik VO + görselleştirme
├── tests/
├── docs/
├── assets/
│   ├── demo.gif              # README: tüm çekimin annotate GIF’i
│   └── plot.png              # README: yörünge özeti
├── README.md
├── pyproject.toml
└── requirements.txt
```

> 📦 Ham video ve koşu çıktıları `.gitignore`’da. README görselleri `assets/` istisnası.

---

<div align="center">
<sub>Bilgisayarla görü / monoküler metrik görsel odometri üzerine bir araştırma çalışması. V1.</sub>
</div>
