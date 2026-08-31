# Ofis videosundan 3D harita — Tasarım

**Tarih:** 2026-08-31
**Durum:** Taslak — onay bekliyor
**Kapsam:** DCMA (Distance_on_Video / Depth-Based-Camera-Motion-Analyzer) üzerine, yürüyen el kamerası videosundan **metre ölçekli 3D ofis haritası**.
**Bu spec’in uygulama planı:** yalnızca **Faz 0 + Faz 1**. VGGT / döngü kapatma kendi planını alır.

---

## 1. Problem

DCMA V2 videodan şunları üretiyor:

- kareler + `K` (`manifest`)
- metrik derinlik (`depth_cache/`, DA2 Indoor-Large)
- `T_wc` yörünge (`trajectory.json`)
- kuş bakışı **2D occupancy** (10 cm xz, kamera yüksekliği dilimi)

Occupancy bir kroki: CAD değil, döngü kapatma yok, geri dönünce duvar kayabilir.

`AI_3D_env` tek fotoğraftan oda mesh’i çıkarıyor. Tek karede görünmeyen yüzey yok; açı değişince boşluk fiziken doğru. Ofis videosunda o yüzeyler **başka karelerde duruyor**. Kör noktayı üretken modelle uydurmak bu üründe yanlış araç.

Nihai amaç: ofisi gezen kişinin videosundan **ofisin 3D haritası**.

## 2. Proje sınırı (ayrı geliştirme)

| Proje | Rol |
|---|---|
| **DCMA** | Video → poz + derinlik → **3D harita**. Bu spec burada yaşar. |
| **AI_3D_env** | Tek görüntü deneyleri (TripoSR obje, DA2 `world.obj`). DCMA’ya import edilmez, kopyalanmaz. |

Paylaşılan fikir (kod değil): derinlik + `K` → 3B, OpenCV `+x` sağ / `+y` aşağı / `+z` ileri.

Kapsam dışı (bu spec ve Faz 0–1):

- FlashWorld / Marble / sahte yeni açı
- TripoSR, obje izolasyonu, Grounding DINO
- Texture / PBR
- Gerçek zamanlı
- GPS / kat planı CAD
- VGGT, MASt3R, COLMAP, döngü kapatma (Faz 2+; aşağıda yol haritası)

## 3. Hedef

Mevcut bir DCMA koşusundan (`Result/<ad>/`) **yeni model çalıştırmadan**:

1. Her VO keyframe derinliğini o karenin `T_wc` ile dünya çerçevesine bas.
2. Voxel ile seyrelt.
3. `map.ply` yaz (xyz + RGB). CloudCompare / Blender’da açı değiştirince ofis dolu görünsün.

Kabul (sentetik): bilinen koridor + 2 m ileri kamera → birleşik bulutta ikinci duvar dilimi `+z` yönünde ~2 m uzamış; GPU yok.

Kabul (gerçek, ölçüm değil gözlem): mevcut ofis videosu üzerinde ilk bakışta duvar/mobilya tanınır; drift çift duvar yaparsa bu **bilinen sınırlama**, Faz 2 gerekçesi.

## 4. Mimari

```
Mevcut DCMA koşusu (değişmez)
  Result/<ad>/
    frames/*.png
    depth_cache/{i:06d}.npy
    trajectory.json     # poses[k], steps[k].frame_from
    occupancy.npz       # 2D kroki; kalır

Yeni, çevrimdışı:
  python -m dcma.map.build --run Result/<ad> --voxel 0.03
    → Result/<ad>/map.ply
    → Result/<ad>/map_meta.json
```

VO CLI’sine harita **gömülmez**. Derinlik zaten önbellekte; harita parametresi (voxel) VO’yu tekrar koşturmamalı.

```
src/dcma/map/
  occupancy.py   # mevcut 2D
  fuse.py        # derinlik+RGB → dünya noktaları, voxel
  ply.py         # PLY yaz
  build.py       # CLI: koşu dizininden map.ply
  poses.py       # trajectory.json → frame_idx → T_wc
```

`dcma.vo.pose.backproject` seyrek özellik içindir. Harita **düzenli ızgara** unproject eder (`occupancy.splat` gibi, ama y dilimi yok, RGB var, çıktı 3B).

### 4.1 Poz–kare eşlemesi (kritik)

`cli.py` sırası:

```
occ.splat(..., traj.poses[-1], frame_idx=i)   # henüz add_step yok
traj.add_step(..., frame_from=i, frame_to=j)
```

- `poses[0]` = dünya = ilk keyframe, `T = I`
- `steps[k].frame_from` karesinin dünyası `poses[k]`
- `steps[k].frame_to` karesinin dünyası `poses[k+1]`

Harita: her `steps[k]` için derinlik `frame_from` + `poses[k]`. Off-by-one çift ofis üretir; sentetik test bunu kilitler.

`depth_cache` anahtarı `f"{i:06d}"` — `frame_from` ile aynı.

### 4.2 Unproject

Occupancy ile aynı pinhole (`intrinsics.py`):

```
z = depth[v, u]
x = (u - cx) * z / fx
y = (v - cy) * z / fy
X_cam = (x, y, z, 1)
X_world = T_wc @ X_cam
```

Geçerli: `min_depth < z < max_depth` (indoor 0.3–15 m). Voxel: `floor(xyz / voxel)`, hücrede ilk nokta (renk ortalama YAGNI).

### 4.3 Çıktı

| Dosya | İçerik |
|---|---|
| `map.ply` | ASCII xyz + uchar RGB; OpenCV `+y` aşağı dünya (ilk kare). Blender’da gerekirse `Y` çevir. |
| `map_meta.json` | voxel, n_points, n_frames, min/max xyz, derinlik aralığı |

2D occupancy ve annotate video **aynı kalır**.

## 5. Fazlar

| Faz | İçerik | Çıkış |
|---|---|---|
| **0** | `fuse.py` + `ply.py` + poz indeksi; sentetik koridor | GPU’suz test yeşil |
| **1** | `dcma.map.build` mevcut `Result/` üzerinde; README | Ofis videosundan `map.ply` |
| **2** | VGGT/MASt3R global poz; DA2 ölçeğine hizala; tekrar fuse | Drift/çift duvar azalır |
| **3** | TSDF/Poisson mesh (`map.obj`) | Blender mesh |
| **4** | Döngü kapatma / pose-graph | Tur atınca duvar tek |

Faz 2+ ayrı plan. Ölçülmemiş birleşik bulut üzerine VGGT varsayımı konmaz.

## 6. Bilinen sınırlamalar (Faz 1 sonunda README)

- Harita VO drift’ini miras alır; döngü kapatma yok.
- DA2 kare kare tutarsız olabilir (aynı duvar kalınlaşır).
- Cadde/cam/düz beyaz duvar VO’yu zaten bozar; harita da bozulur.
- Görülmeyen oda (hiç girilmeyen) boş kalır — bu doğru.
- Obje ayrımı yok: tek sahne bulutu.

## 7. Test

Occupancy’deki sentetik koridor (`x = ±1 m` duvar) + ikinci poz `T[2,3] = 2`:

- Birleşik `z` yayılımı ~2 m artar
- Voxel sonrası nokta sayısı ham piksel sayısından küçük
- `map.ply` okununca xyz boyutu meta ile uyumlu
- `poses[k]` ↔ `frame_from` yanlış bağlanırsa test **bilerek** fail (yanlış indeksi kilitle)

GPU / VGGT bu planda yok.
