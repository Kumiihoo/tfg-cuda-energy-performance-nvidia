#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
OUT="${2:-results/a100/energy/power.csv}"
MS="${3:-100}"
SMI_PID=""

mkdir -p "$(dirname "$OUT")"

cleanup() {
  if [[ -n "${SMI_PID:-}" ]] && kill -0 "$SMI_PID" >/dev/null 2>&1; then
    kill "$SMI_PID" >/dev/null 2>&1 || true
    wait "$SMI_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM HUP

LC_ALL=C LANG=C nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,pstate,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader,nounits -lms "$MS" > "$OUT" &
SMI_PID=$!
wait "$SMI_PID"

