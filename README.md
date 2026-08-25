<div align="center">

# 🎥 Depth-Based 6-Direction Motion Analyzer

**Tek bir monoküler videodan, hiçbir sensör verisi olmadan kamera hareket yönünü çıkarır.**

Her kare için [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-base-hf)
ile derinlik haritası üretilir; ardışık karelerin **bölgesel derinlik değişimi** karşılaştırılarak
kameranın 6 yönden hangisine hareket ettiği belirlenir ve sonuç doğrudan videoya gömülür.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
![Status](https://img.shields.io/badge/status-araştırma%20prototipi-orange)

</div>

---

## 🎬 Demo

> 🚧 **Yeni demo çekimi hazırlanıyor.** Daha kontrollü bir çekim yapıldıktan sonra
> annotate edilmiş demo GIF'i buraya eklenecek.

<!-- Yeni video hazır olduğunda:
     ./Scripts/make_demo_gif.sh out/annotated_6_direction.mp4 assets/demo.gif 12 640
     ve aşağıdaki satırın yorumunu kaldır: -->
<!-- <div align="center"><img src="assets/demo.gif" width="640" alt="6 yönlü hareket analizi demosu"></div> -->

Videonun sağ alt köşesinde o anki hareket yönü renk kodlu olarak, sol üstte kare numarası,
sol altta ise yön dağılımı görünür.

---

## 📌 Ne Yapar?

Klasik yaklaşımlar kamera hareketini **optik akış** ile (piksellerin 2B kayması) tahmin eder.
Bu yöntemin temel zaafı, *sahnedeki nesne hareketi* ile *kameranın kendi hareketini*
ayırt edememesidir. Bu proje bunun yerine **sahnenin 3B yapısındaki değişime** bakar:

> Kamera ileri giderse **her yer** yakınlaşır. Kamera sağa kayarsa sol taraf uzaklaşırken
> sağ taraf yakınlaşır. Bu asimetri, derinlik haritasının bölgesel istatistiklerinde okunabilir.

### Tespit Edilen Yönler

| Yön | Ekrandaki Renk | Anlamı |
|:---|:---|:---|
| **İLERİ** | 🟢 Yeşil | Kamera sahneye yaklaşıyor |
| **GERİ** | 🔴 Kırmızı | Kamera sahneden uzaklaşıyor |
| **SAĞA** | 🔵 Mavi | Kamera sağa öteleniyor |
| **SOLA** | 🩵 Cyan | Kamera sola öteleniyor |
| **YUKARI** | 🟣 Magenta | Kamera yukarı öteleniyor |
| **AŞAĞI** | 🟡 Sarı | Kamera aşağı öteleniyor |
| **BELİRSİZ** | ⚪ Beyaz | Baskın bir yön kuralı tetiklenmedi |

> ℹ️ Bunlar **3 eksen × 2 yön = 6 öteleme yönüdür.** Dönme (pan / tilt / roll) şu an
> tespit edilmiyor — bkz. [Bilinen Sınırlamalar](#-bilinen-sınırlamalar).

---

## ⚙️ Nasıl Çalışır?

```
   Video                 Kareler              Depth Anything V2         3×3 Bölgesel
 (.mp4)      ──────▶   Frame_0001.jpg  ──────▶   derinlik haritası  ──────▶   ortalamalar
                       Frame_0002.jpg               (H×W float)              (9 değer)
                            ...                                                  │
                                                                                 ▼
   Annotate                  Yön kuralları              Δ = kare[i+5] − kare[i]
   edilmiş     ◀──────    (eşik = 0.01)      ◀──────    bölgesel fark matrisi
    video
```

### 1️⃣ Derinlik Tahmini
Her kare `depth-anything/Depth-Anything-V2-base-hf` modeline verilir; model piksel başına
bir derinlik değeri üretir. GPU varsa `accelerate` üzerinden otomatik seçilir.

### 2️⃣ 3×3 Bölgesel Özetleme
Derinlik haritası 9 bölgeye bölünür ve her bölgenin **ortalaması** alınır. Bu adım gürültüyü
bastırır ve haritayı 9 sayıya indirir.

```
┌─────────┬─────────┬─────────┐
│  (0,0)  │  (0,1)  │  (0,2)  │  ◀── üst satır   → YUKARI / AŞAĞI sinyali
├─────────┼─────────┼─────────┤
│  (1,0)  │  (1,1)  │  (1,2)  │
├─────────┼─────────┼─────────┤
│  (2,0)  │  (2,1)  │  (2,2)  │  ◀── alt satır
└─────────┴─────────┴─────────┘
     ▲                   ▲
 sol sütun            sağ sütun  → SAĞA / SOLA sinyali
```

### 3️⃣ Zamansal Fark
`kare[i]` ile `kare[i+5]` karşılaştırılır. **5 kare aralığı** bilinçli bir seçimdir: ardışık
kareler arasındaki hareket, modelin tahmin gürültüsünün altında kalacak kadar küçüktür.

### 4️⃣ Yön Kuralları
`Δ = bölge_ortalamaları[i+5] − bölge_ortalamaları[i]` matrisi üzerinde, ilk eşleşen kural kazanır:

| Koşul (eşik `τ = 0.01`) | Sonuç |
|:---|:---|
| **9 bölgenin tamamında** `Δ < −τ` | **İLERİ** |
| **9 bölgenin tamamında** `Δ > +τ` | **GERİ** |
| `mean(sol sütun) > τ` **ve** `mean(sağ sütun) < −τ` | **SAĞA** |
| `mean(sol sütun) < −τ` **ve** `mean(sağ sütun) > τ` | **SOLA** |
| `mean(üst satır) > τ` **ve** `mean(alt satır) < −τ` | **YUKARI** |
| `mean(üst satır) < −τ` **ve** `mean(alt satır) > τ` | **AŞAĞI** |
| yukarıdakilerin hiçbiri | **BELİRSİZ** |

### 5️⃣ Video Annotation
Bulunan yön etiketi, OpenCV ile orijinal videonun üzerine yazılır. Codec olarak sırayla
`H264 → XVID → MP4V → MJPG` denenir; sistemde çalışan ilk codec kullanılır.

---

## 🚀 Kurulum

```bash
git clone https://github.com/Erenosxx/Depth-Based-6-DoF-Motion-Analyzer.git
cd Depth-Based-6-DoF-Motion-Analyzer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 💡 `torch`'u kendi CUDA sürümünüze uygun kurmanız önerilir:
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

Model ilk çalıştırmada Hugging Face'den otomatik indirilir (~200 MB, base checkpoint).

---

## 🧪 Kullanım

### 1. Video karelerini çıkar

```bash
mkdir -p Video_Frames
ffmpeg -i girdi.mp4 Video_Frames/Frame_%04d.jpg
```

### 2. Yolları ayarla

[`Scripts/Distance_mesurement_4_2.py`](Scripts/Distance_mesurement_4_2.py) içindeki üç
değişkeni kendi ortamınıza göre düzenleyin:

```python
video_path        = "/yol/girdi.mp4"            # annotate edilecek orijinal video
frames_path       = "/yol/Video_Frames"         # çıkarılmış JPG kareler
output_video_path = "/yol/out/annotated.mp4"    # üretilecek video
```

### 3. Çalıştır

```bash
python Scripts/Distance_mesurement_4_2.py
```

Script sırayla: kareleri yükler → her kare çifti için derinlik tahmini yapar →
yön analizini uygular → annotate edilmiş videoyu yazar → çıktıyı doğrular.

### 4. (İsteğe bağlı) README için demo GIF üret

```bash
./Scripts/make_demo_gif.sh out/annotated.mp4 assets/demo.gif 12 640
```

---

## 📤 Çıktı

Konsolda yön dağılımı özeti yazdırılır:

```
📊 Hareket İstatistikleri:
     İLERİ:  12 frame ( 5.6%)
      GERİ:  18 frame ( 8.5%)
      SAĞA:  32 frame (15.0%)
      SOLA:   5 frame ( 2.3%)
    YUKARI:   2 frame ( 0.9%)
     AŞAĞI:   6 frame ( 2.8%)
  BELİRSİZ: 138 frame (64.8%)
```

<sub>Yukarıdaki değerler örnek bir çalıştırmadandır. Yüksek **BELİRSİZ** oranının nedeni
aşağıda açıklanıyor.</sub>

**Referans çalıştırma:** 1440×1440 @ 30 fps, ~7 saniye, 207 kare.

---

## 🔬 Teknik Detaylar

| Bileşen | Değer |
|:---|:---|
| Derinlik modeli | `depth-anything/Depth-Anything-V2-base-hf` |
| Framework | PyTorch + 🤗 Transformers (`pipeline("depth-estimation")`) |
| Cihaz seçimi | `accelerate.get_backend()` ile otomatik (CUDA / MPS / CPU) |
| Video I/O | OpenCV (`cv2.VideoCapture` / `VideoWriter`) |
| Grid | 3×3 (9 bölge), bölge özeti = aritmetik ortalama |
| Kare karşılaştırma aralığı | 5 kare |
| Eşik değeri `τ` | 0.01 |
| Denenen codec'ler | H264, XVID, MP4V, MJPG (ilk çalışan seçilir) |

---

## ⚠️ Bilinen Sınırlamalar

Bu bir **araştırma prototipidir**. Şeffaflık için bilinen zayıf noktalar:

| # | Sınırlama | Neden / Etki |
|:--|:---|:---|
| 1 | **Yüksek BELİRSİZ oranı** | İLERİ/GERİ kuralı **9 bölgenin tamamının** aynı yönde değişmesini şart koşar. Tek bir aykırı bölge bile kuralı düşürür. |
| 2 | **Göreli (relative) derinlik** | Depth Anything V2 metrik değil, ölçek ve kayması kareden kareye değişebilen *affine-invariant* derinlik üretir. Kareler arası mutlak fark bu yüzden gürültülüdür — sinyalin bir kısmı gerçek hareket değil, ölçek kaymasıdır. |
| 3 | **Dönme tespiti yok** | Yalnızca 3 eksende öteleme sınıflandırılır; pan / tilt / roll ölçülmez. |
| 4 | **Yön var, büyüklük yok** | Çıktı kategorik bir etikettir; hız veya kat edilen mesafe üretilmez. |
| 5 | **Gereksiz hesap** | Döngü her adımda `kare[i]` ve `kare[i+5]` için ayrı ayrı çıkarım yapar; böylece neredeyse her kare **iki kez** modele girer. Derinlik haritalarını önbelleğe almak süreyi ~2× kısaltır. |
| 6 | **Sabit kodlanmış yollar** | Yollar script içinde tanımlı; henüz CLI argümanı yok. |
| 7 | **Ekrandaki sayaçlar statik** | Sol alttaki dağılım, analiz bittikten sonra hesaplanan **nihai toplamlardır**; her karede aynı değerleri gösterir (kümülatif ilerlemeyi değil). |
| 8 | **Kare hizalaması** | Etiket, `kare[i] → kare[i+5]` aralığını tanımlar ama karenin *kendisine* yazılır; ayrıca çıkarılan JPG sayısı ile video kare sayısı birebir olmayabilir. |

---

## 🗺️ Yol Haritası

- [ ] Yeni, kontrollü demo çekimi + README'ye demo GIF
- [ ] Derinlik haritası önbelleği (~2× hızlanma)
- [ ] `argparse` ile CLI arayüzü
- [ ] Kare başına ölçek/kayma normalizasyonu (madde 2'yi azaltmak için)
- [ ] Yumuşatılmış karar mekanizması (katı `all()` yerine oy / skor tabanlı)
- [ ] Dönme (pan / tilt / roll) tespiti
- [ ] Etiketlenmiş referans video ile nicel doğruluk ölçümü

---

## 🧭 Projenin Evrimi

Sonuçtaki yaklaşım, dört ayrı denemenin ardından ortaya çıktı:

| Deneme | Yaklaşım | Neden yetersiz kaldı |
|:--|:---|:---|
| **1** | Lucas-Kanade optik akış + grid tabanlı nokta takibi (Gradio arayüzü, V1→V3) | 2B piksel kayması, kamera hareketini sahne hareketinden ayırt edemedi |
| **2** | `KameraHareketTespiti` sınıfı: 6 yön + zoom, kümülatif referans değerler, JSON çıktı | Hâlâ optik akış tabanlı; eşikler sahneye aşırı duyarlıydı |
| **3** | Kareleri JPG olarak dışa aktarma + ardışık kare analizi | Analiz altyapısı hazırlandı, yöntem değişmedi |
| **4** | **Depth Anything V2'ye geçiş.** İlk olarak 2 yön (İleri/Geri), sonra 3×3 bölgesel analiz ile 6 yön + videoya gömme | ✅ Mevcut yaklaşım |

Kilit içgörü: **2B hareketi tahmin etmeye çalışmak yerine 3B yapıyı çıkarıp onun değişimine bakmak.**

---

## 📂 Proje Yapısı

```
Depth-Based-6-DoF-Motion-Analyzer/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                              # README görselleri (demo GIF buraya)
└── Scripts/
    ├── Distance_mesurement_4_2.py       # Ana pipeline: derinlik → 6 yön → annotate
    └── make_demo_gif.sh                 # Çıktı videosundan README GIF'i üretir
```

> 📦 Ham videolar ve çıkarılmış kareler `.gitignore` ile hariç tutulmuştur
> (GitHub'ın 100 MB dosya limitini aşıyorlar).

---

<div align="center">
<sub>Bilgisayarla görü / monoküler derinlik tahmini üzerine bir araştırma çalışması.</sub>
</div>
