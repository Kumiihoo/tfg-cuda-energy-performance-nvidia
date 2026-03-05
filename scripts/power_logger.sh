#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
OUT="${2:-results/a100/energy/power.csv}"
MS="${3:-100}"

mkdir -p "$(dirname "$OUT")"

LC_ALL=C LANG=C nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,pstate,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader,nounits -lms "$MS" > "$OUT"


