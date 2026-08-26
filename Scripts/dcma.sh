#!/usr/bin/env bash
# local.env içindeki DCMA_PYTHON ile çalıştırır (dosya git'e girmez).
# Örnek: ./Scripts/dcma.sh -m dcma.cli --video Data/girdi.mp4 --out Result/calisma_adi ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/local.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/local.env"
  set +a
fi

PYTHON="${DCMA_PYTHON:-python3}"
exec env -u VIRTUAL_ENV -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT/src" \
  "$PYTHON" "$@"
