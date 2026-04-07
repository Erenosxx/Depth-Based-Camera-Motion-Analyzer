# 6 Yönlü Kamera Hareket Analizi ve Video Annotation Sistemi

Depth Anything V2 derinlik tahmin modeli kullanarak video karelerindeki kamera hareketini **6 yönde** (İleri, Geri, Sağa, Sola, Yukarı, Aşağı) tespit eden ve sonuçları doğrudan video üzerine yazan bir bilgisayarla görü projesidir.

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Projenin Evrimi](#projenin-evrimi)
- [Nasıl Çalışır?](#nasıl-çalışır)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Çıktı](#çıktı)
- [Proje Yapısı](#proje-yapısı)
- [Teknik Detaylar](#teknik-detaylar)

---

## Proje Özeti

Bu sistem, bir videodan çıkarılmış kareleri **Depth Anything V2** modeline besleyerek her kare için bir derinlik haritası üretir. Ardından ardışık karelerin derinlik haritalarını karşılaştırarak kameranın hangi yöne hareket ettiğini belirler. Tespit edilen hareket yönü, renk kodlarıyla birlikte orijinal videoya yazılarak görselleştirilir.

### Tespit Edilen Yönler

| Yön | Renk | Açıklama |
|---|---|---|
| **İLERİ** | 🟢 Yeşil | Kamera sahneye doğru yaklaşıyor |
| **GERİ** | 🔴 Kırmızı | Kamera sahneden uzaklaşıyor |
| **SAĞA** | 🔵 Mavi | Kamera sağa kayıyor |
| **SOLA** | 🔵 Cyan | Kamera sola kayıyor |
| **YUKARI** | 🟣 Magenta | Kamera yukarı hareket ediyor |
| **AŞAĞI** | 🟡 Sarı | Kamera aşağı hareket ediyor |
| **BELİRSİZ** | ⚪ Beyaz | Belirgin bir hareket yok |

---

## Projenin Evrimi

Bu proje, birden fazla iterasyon (deneme) üzerinden geliştirilmiştir:

### Deneme 1 — Optik Akış ile Nokta Takibi
- Lucas-Kanade optik akış algoritması kullanılarak ilk denemeler yapıldı.
- Video üzerinde belirli noktalar seçilip izlendi (Gradio arayüzü ile interaktif kullanım).
- Grid tabanlı özellik takibi sistemi geliştirildi (V1 → V2 → V3).

### Deneme 2 — Kamera Hareket Tespiti Sınıfı
- `KameraHareketTespiti` sınıfı ile 6 yön + zoom algılama.
- Referans değer sistemi (kümülatif ileri/geri, sağ/sol, yukarı/aşağı).
- Sonuçlar JSON formatında kaydedildi.

### Deneme 3 — Video Kare Çıkarma ve Analiz
- Video karelerini JPG dosyası olarak dışa aktarma.
- Ardışık kare analizi ve ilerleme değeri takibi.

### Deneme 4 — Derinlik Tahmini ile Hareket Analizi (Mevcut)
- **Depth Anything V2** modeline geçiş.
- İlk olarak 2 yön (İleri/Geri), ardından 6 yöne genişletildi.
- 3×3 bölgesel analiz ile yön tespiti.
- Sonuçların videoya gömülmesi (annotation).

---

## Nasıl Çalışır?

### 1. Derinlik Haritası Üretimi
Her video karesi, Hugging Face üzerindeki `depth-anything/Depth-Anything-V2-base-hf` modeline gönderilir. Model, her piksel için bir derinlik değeri tahmin eder.

### 2. Bölgesel Analiz (3×3 Grid)
Derinlik haritası 3×3'lük 9 bölgeye ayrılır. Her bölgenin ortalama derinlik değeri hesaplanır.

```
┌─────────┬─────────┬─────────┐
│ (0,0)   │ (0,1)   │ (0,2)   │  ← Üst satır
├─────────┼─────────┼─────────┤
│ (1,0)   │ (1,1)   │ (1,2)   │  ← Orta satır
├─────────┼─────────┼─────────┤
│ (2,0)   │ (2,1)   │ (2,2)   │  ← Alt satır
└─────────┴─────────┴─────────┘
  Sol sütun  Orta      Sağ sütun
```

### 3. Kare Karşılaştırması
5 kare arayla iki derinlik haritası karşılaştırılır (`frame[i]` ↔ `frame[i+5]`). Bölgesel derinlik değişimleri hesaplanır.

### 4. Yön Belirleme Kuralları

| Koşul | Sonuç |
|---|---|
| Tüm bölgelerde derinlik azaldı | **İLERİ** (kamera yaklaşıyor) |
| Tüm bölgelerde derinlik arttı | **GERİ** (kamera uzaklaşıyor) |
| Sol sütun arttı + Sağ sütun azaldı | **SAĞA** |
| Sol sütun azaldı + Sağ sütun arttı | **SOLA** |
| Üst satır arttı + Alt satır azaldı | **YUKARI** |
| Üst satır azaldı + Alt satır arttı | **AŞAĞI** |
| Diğer durumlar | **BELİRSİZ** |

### 5. Video Annotation
Tespit edilen hareket yönü, orijinal video üzerine yazılır:
- **Sağ alt köşe:** Anlık hareket yönü (renk kodlu)
- **Sol üst köşe:** Kare numarası
- **Sol alt köşe:** Tüm yönler için toplam sayaçlar

---

## Kurulum

### Gereksinimler

```bash
pip install torch transformers accelerate pillow opencv-python numpy
```

### Model

İlk çalıştırmada `depth-anything/Depth-Anything-V2-base-hf` modeli Hugging Face'den otomatik olarak indirilir. GPU varsa otomatik olarak kullanılır.

---

## Kullanım

### 1. Video Karelerini Hazırlama

Kaynak video karelerinin JPG formatında bir klasörde bulunması gerekir. (Varsayılan: `Original_Video/Video_Frames/`)

### 2. Script'i Çalıştırma

Dosya içindeki yolları kendi ortamınıza göre düzenleyin:

```python
video_path = "/path/to/original_video.mp4"
frames_path = "/path/to/Video_Frames"
output_video_path = "/path/to/output/annotated_6_direction.mp4"
```

Ardından çalıştırın:

```bash
python Distance_mesurement_4_2.py
```

### 3. İşlem Adımları

Script sırasıyla şunları yapar:
1. Frame dosyalarını sıralı olarak yükler (JPG)
2. Her kare çifti için Depth Anything V2 ile derinlik tahmini yapar
3. 6 yönlü hareket analizi gerçekleştirir
4. Sonuçları orijinal video üzerine yazarak annotated video oluşturur
5. Video dosyasını test eder ve istatistikleri yazdırır

---

## Çıktı

### Annotated Video
- Hareket yönü renk kodlu olarak sağ alt köşede gösterilir
- Kare numarası sol üst köşede yer alır
- Kümülatif hareket istatistikleri sol alt köşede listelenir

### Konsol Çıktısı (Örnek)

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

---

## Proje Yapısı

```
Eren_Deneme/
├── Deneme_1/                   # Optik akış ve grid tabanlı takip denemeleri
│   ├── Auto_Grid/              # Grid tabanlı özellik takibi V1
│   ├── Auto_Grid_V2/           # V2
│   ├── Auto_Grid_V3/           # V3 (Final dahil)
│   └── deneme_1_optic_flow/    # Gradio arayüzlü nokta takibi
│
├── Deneme_2/                   # Lucas-Kanade optik akış ile kamera hareket tespiti
│   ├── Distance_mesurement_2.py
│   └── *.json                  # Analiz sonuçları
│
├── Deneme_3/                   # Video kare çıkarma ve analiz
│   └── Distance_mesurement_3.py
│
├── deneme_4/                   # Depth Anything V2 ile derinlik tabanlı analiz
│   ├── İleri-Geri/             # 2 yönlü tespit (İleri/Geri)
│   │   ├── Distance_mesurement_4.py
│   │   └── Distance_mesurement_4_1.py
│   └── İleri-Geri-Sağa-Sola/  # 6 yönlü tespit ← (Bu klasör)
│       ├── Distance_mesurement_4_2.py   # Ana script
│       └── README.md
│
└── Original_Video/             # Kaynak video ve yardımcı araçlar
    ├── 20250828_102115.mp4     # Orijinal video
    ├── Reverse.py              # Video tersine çevirme
    └── Video_Frames/           # Çıkarılmış kareler
```

---

## Teknik Detaylar

| Bileşen | Detay |
|---|---|
| **Derinlik Modeli** | Depth Anything V2 Base (`depth-anything/Depth-Anything-V2-base-hf`) |
| **Framework** | PyTorch + Hugging Face Transformers |
| **GPU Desteği** | Accelerate kütüphanesi ile otomatik cihaz seçimi |
| **Video İşleme** | OpenCV (cv2) |
| **Kare Karşılaştırma Aralığı** | 5 kare |
| **Eşik Değeri (threshold)** | 0.01 |
| **Grid Boyutu** | 3×3 (9 bölge) |
| **Desteklenen Codec'ler** | H264, XVID, MP4V, MJPG |
