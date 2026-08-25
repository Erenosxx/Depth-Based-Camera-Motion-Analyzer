#!/usr/bin/env python3
"""
[ESKİ / LEGACY] 6 Yönlü Hareket Analizi ve Video Annotation Sistemi

Bu script, projenin 4. deneme aşamasıdır ve tarihsel referans olarak korunuyor.
Göreli (affine-invariant) derinlik haritalarının 3×3 bölge ortalamalarını
karşılaştırıp kategorik bir yön etiketi üretir; metre üretmez.

Yerine geçen: src/dcma/ altındaki metrik görsel odometri pipeline'ı.
Gerekçe: docs/superpowers/specs/2026-08-25-metric-6dof-vo-design.md
"""

from transformers import pipeline
import torch
from accelerate.test_utils.testing import get_backend
from PIL import Image
import os
import glob
import cv2
import numpy as np

# Device setup
device, _, _ = get_backend()
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint, device=device)

def analyze_depth_movement(predictions, predictions2, threshold=0.01):
    """6 yönlü hareket analizi"""
    if isinstance(predictions["predicted_depth"], torch.Tensor):
        depth_map1 = predictions["predicted_depth"].squeeze().cpu().numpy()
    else:
        depth_map1 = predictions["predicted_depth"]
 
    if isinstance(predictions2["predicted_depth"], torch.Tensor):
        depth_map2 = predictions2["predicted_depth"].squeeze().cpu().numpy()
    else:
        depth_map2 = predictions2["predicted_depth"]
 
    h, w = depth_map1.shape
 
    grid_size = 3
    region_h = h // grid_size
    region_w = w // grid_size
 
    region_means1 = np.zeros((grid_size, grid_size))
    region_means2 = np.zeros((grid_size, grid_size))
 
    for i in range(grid_size):
        for j in range(grid_size):
 
            start_h = i * region_h
            end_h = (i + 1) * region_h if i < grid_size - 1 else h
            start_w = j * region_w
            end_w = (j + 1) * region_w if j < grid_size - 1 else w
 
            region1 = depth_map1[start_h:end_h, start_w:end_w]
            region2 = depth_map2[start_h:end_h, start_w:end_w]
 
            region_means1[i, j] = np.mean(region1)
            region_means2[i, j] = np.mean(region2)
 
    depth_changes = region_means2 - region_means1
 
    movement_direction = analyze_regional_changes(depth_changes, threshold)
 
    return movement_direction, region_means1, region_means2, depth_changes
 
def analyze_regional_changes(depth_changes, threshold):
    """Bölgesel değişimleri analiz et"""
    all_decreased = np.all(depth_changes < -threshold)
    all_increased = np.all(depth_changes > threshold)
 
    left_changes = depth_changes[:, 0]  
    right_changes = depth_changes[:, 2]  
    top_changes = depth_changes[0, :]   
    bottom_changes = depth_changes[2, :]
 
    if all_decreased:
        return "İLERİ"
    elif all_increased:
        return "GERİ"
    elif np.mean(left_changes) > threshold and np.mean(right_changes) < -threshold:
        return "SAĞA"
    elif np.mean(left_changes) < -threshold and np.mean(right_changes) > threshold:
        return "SOLA"
    elif np.mean(top_changes) > threshold and np.mean(bottom_changes) < -threshold:
        return "YUKARI"
    elif np.mean(top_changes) < -threshold and np.mean(bottom_changes) > threshold:
        return "AŞAĞI"
    else:
        return "BELİRSİZ"

# Video ve frame path'leri - kendi ortaminiza gore duzenleyin
# frames_path icindeki JPG kareleri once cikarin:
#   ffmpeg -i girdi.mp4 Video_Frames/Frame_%04d.jpg
video_path = "path/to/input.mp4"                      # annotate edilecek orijinal video
frames_path = "path/to/Video_Frames"                  # cikarilmis JPG kareler
output_video_path = "path/to/out/annotated_6_direction.mp4"   # uretilecek video

print("🎥 6 Yönlü Hareket Analizi ve Video Annotation")
print("="*60)

# JPG dosyalarını sıralı şekilde al
frame_files = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
print(f"📁 Bulunan frame sayısı: {len(frame_files)}")

if len(frame_files) == 0:
    print("❌ Frame dosyası bulunamadı!")
    exit(1)

# Frame'leri PIL Image olarak yükle
print("\n📂 Frame'ler yükleniyor...")
video_frames = []
for i, frame_file in enumerate(frame_files):
    try:
        img = Image.open(frame_file)
        video_frames.append(img)
        if i % 25 == 0:
            print(f"  Yüklendi: {i+1}/{len(frame_files)} - {os.path.basename(frame_file)}")
    except Exception as e:
        print(f"❌ Hata: {frame_file} - {e}")

print(f"✅ Toplam yüklenen frame: {len(video_frames)}")

# Hareket analizi
print("\n🔍 6 Yönlü hareket analizi yapılıyor...")
movement_results = []
movement_counts = {
    "İLERİ": 0, "GERİ": 0, "SAĞA": 0, 
    "SOLA": 0, "YUKARI": 0, "AŞAĞI": 0, "BELİRSİZ": 0
}

for i in range(len(video_frames)-5):
    if i % 10 == 0:
        progress = (i / (len(video_frames)-5)) * 100
        print(f"  İşleniyor: {progress:.1f}% - Frame {i+1}/{len(video_frames)-5}")

    image_1 = video_frames[i]
    image_2 = video_frames[i+5]

    try:
        predictions = pipe(image_1)
        predictions2 = pipe(image_2)

        movement, means1, means2, changes = analyze_depth_movement(predictions, predictions2)
        movement_results.append(movement)
        movement_counts[movement] += 1
        
    except Exception as e:
        print(f"⚠️ Frame {i+1} analiz hatası: {e}")
        movement_results.append("BELİRSİZ")
        movement_counts["BELİRSİZ"] += 1

# Son 5 frame için "BELİRSİZ" ekle
for i in range(5):
    movement_results.append("BELİRSİZ")
    movement_counts["BELİRSİZ"] += 1

print(f"✅ Hareket analizi tamamlandı!")
print(f"\n📊 Hareket İstatistikleri:")
for direction, count in movement_counts.items():
    percentage = (count / len(movement_results)) * 100
    print(f"  {direction:>8}: {count:>3} frame ({percentage:>5.1f}%)")

# Video oluşturma
print(f"\n🎬 Annotated video oluşturuluyor...")

# Orijinal videoyu aç
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Video açılamadı: {video_path}")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"📺 Video özelikleri: {width}x{height} @ {fps} FPS")

# Output klasörünü oluştur
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Video writer oluştur - Uyumlu codec'ler dene
fourcc_options = [
    ('H264', '.mp4'),
    ('XVID', '.avi'), 
    ('MP4V', '.mp4'),
    ('MJPG', '.avi')
]

out = None
final_output_path = ""

for codec, ext in fourcc_options:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_path = output_video_path.replace('.mp4', ext)
        out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            final_output_path = test_path
            print(f"✅ Video writer başarılı: {codec} codec")
            break
        else:
            out.release()
            
    except Exception as e:
        print(f"⚠️ {codec} codec başarısız: {e}")

if out is None or not out.isOpened():
    print("❌ Hiçbir video codec çalışmadı!")
    cap.release()
    exit(1)

# Hareket renkleri tanımla
movement_colors = {
    "İLERİ": (0, 255, 0),      # Yeşil
    "GERİ": (0, 0, 255),       # Kırmızı
    "SAĞA": (255, 0, 0),       # Mavi
    "SOLA": (255, 255, 0),     # Cyan
    "YUKARI": (255, 0, 255),   # Magenta
    "AŞAĞI": (0, 255, 255),    # Sarı
    "BELİRSİZ": (255, 255, 255) # Beyaz
}

# Video frame'lerini işle
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Hareket bilgisini al
    if frame_index < len(movement_results):
        movement_text = movement_results[frame_index]
    else:
        movement_text = "BELİRSİZ"
    
    # Text özellikleri
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Text rengi
    color = movement_colors.get(movement_text, (255, 255, 255))
    
    # Text boyutunu al
    text_size = cv2.getTextSize(movement_text, font, font_scale, thickness)[0]
    
    # Sağ alt köşe pozisyonu
    text_x = width - text_size[0] - 30
    text_y = height - 30
    
    # Arkaplan dikdörtgeni (okunabilirlik için)
    bg_margin = 15
    cv2.rectangle(frame, 
                  (text_x - bg_margin, text_y - text_size[1] - bg_margin),
                  (text_x + text_size[0] + bg_margin, text_y + bg_margin),
                  (0, 0, 0), -1)
    
    # Ana hareket text'i
    cv2.putText(frame, movement_text, (text_x, text_y), font, font_scale, color, thickness)
    
    # Frame numarası (sol üst köşe)
    frame_info = f"Frame: {frame_index + 1}"
    cv2.putText(frame, frame_info, (20, 40), font, 0.8, (255, 255, 255), 2)
    
    # Hareket sayaçları (sol alt köşe - 6 satır)
    stats_y_start = height - 150
    for i, (direction, count) in enumerate(movement_counts.items()):
        stats_text = f"{direction}: {count}"
        stats_color = movement_colors[direction]
        cv2.putText(frame, stats_text, (20, stats_y_start + i*20), 
                   font, 0.6, stats_color, 2)
    
    # Frame'i videoya yaz
    out.write(frame)
    frame_index += 1
    
    # Progress
    if frame_index % 30 == 0:
        progress = (frame_index / len(movement_results)) * 100
        print(f"  Video: {progress:.1f}% - {frame_index}/{len(movement_results)} frame")

# Temizlik
cap.release()
out.release()

print(f"\n✅ 6 Yönlü annotated video oluşturuldu!")
print(f"📁 Dosya: {final_output_path}")
print(f"\n📊 Final İstatistikler:")
for direction, count in movement_counts.items():
    percentage = (count / len(movement_results)) * 100
    print(f"  {direction:>8}: {count:>3} frame ({percentage:>5.1f}%)")

# Video dosyasını test et
print(f"\n🧪 Video dosyası test ediliyor...")
test_cap = cv2.VideoCapture(final_output_path)
if test_cap.isOpened():
    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    test_cap.release()
    print(f"✅ Video başarıyla oluşturuldu!")
    print(f"  - Süre: {duration:.1f} saniye")
    print(f"  - Frame sayısı: {frame_count}")
else:
    print(f"❌ Video dosyası test edilemedi!")

print("\n🎯 Video oynatma komutları:")
print(f"  mpv '{final_output_path}'")
print(f"  vlc '{final_output_path}'")
print(f"  ffplay '{final_output_path}'")