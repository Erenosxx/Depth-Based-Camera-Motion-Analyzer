#!/usr/bin/env bash
# Annotate edilmiş çıktı videosunun tamamından README için hafif bir GIF üretir.
# Varsayılan: 6× hız + 6 fps + 300 px — ~35 sn’lik çekim ~6 sn / ~35 kare (~2–3 MB).
# GitHub camo ~5 MB üstünü sessizce düşürür; README için bu sınırın altında kalın.
#
# Kullanım:
#   ./Scripts/make_demo_gif.sh <annotated_video> [cikti.gif] [fps] [genislik] [hiz]
#
# Örnek:
#   ./Scripts/make_demo_gif.sh Result/ileri_geri_sol_sag.mp4 assets/demo.gif 6 300 6

set -euo pipefail

IN="${1:?Kullanim: $0 <annotated_video> [cikti.gif] [fps] [genislik] [hiz]}"
OUT="${2:-assets/demo.gif}"
FPS="${3:-6}"
WIDTH="${4:-300}"
SPEED="${5:-6}"

PALETTE="$(mktemp --suffix=.png)"
trap 'rm -f "$PALETTE"' EXIT

mkdir -p "$(dirname "$OUT")"

FILTER="setpts=PTS/${SPEED},fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

ffmpeg -v warning -y -i "$IN" -vf "${FILTER},palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -v warning -y -i "$IN" -i "$PALETTE" \
       -lavfi "${FILTER} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
       "$OUT"

echo "GIF hazir: $OUT ($(du -h "$OUT" | cut -f1))"
echo "5 MB'in altinda tutun: GitHub README ~5 MB ustunu gostermez."
