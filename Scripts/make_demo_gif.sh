#!/usr/bin/env bash
# Annotate edilmiş çıktı videosundan README için hafif bir demo GIF üretir.
#
# Kullanım:
#   ./Scripts/make_demo_gif.sh <annotated_video> [assets/demo.gif] [fps] [genislik]
#
# Örnek:
#   ./Scripts/make_demo_gif.sh out/annotated_6_direction.mp4 assets/demo.gif 12 640

set -euo pipefail

IN="${1:?Kullanim: $0 <annotated_video> [cikti.gif] [fps] [genislik]}"
OUT="${2:-assets/demo.gif}"
FPS="${3:-12}"
WIDTH="${4:-640}"

PALETTE="$(mktemp --suffix=.png)"
trap 'rm -f "$PALETTE"' EXIT

mkdir -p "$(dirname "$OUT")"

FILTER="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

# 1) Video'ya özel optimum palet çıkar  2) o paletle dither'la
ffmpeg -v warning -y -i "$IN" -vf "${FILTER},palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -v warning -y -i "$IN" -i "$PALETTE" \
       -lavfi "${FILTER} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
       "$OUT"

echo "GIF hazir: $OUT ($(du -h "$OUT" | cut -f1))"
echo "10 MB'in altinda tutmak icin fps/genisligi dusurebilirsiniz."
