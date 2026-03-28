#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_compare.sh [options]

Options:
  --a100-root PATH       Root folder for A100 results (default: results/a100)
  --rtx5000-root PATH    Root folder for RTX5000 results (default: results/rtx5000)
  --output-dir PATH      Output folder for compare artifacts (default: results/compare)
  --python BIN           Explicit Python executable
  --help                 Show this help

Examples:
  ./scripts/run_compare.sh
  ./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
EOF
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  if command -v python >/dev/null 2>&1 && python -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1; then
    echo "python"
    return
  fi

  if command -v python3 >/dev/null 2>&1 && python3 -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo ""
}

require_python3() {
  local bin="$1"
  if ! "$bin" -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1; then
    echo "Error: '$bin' must resolve to a Python 3 interpreter" >&2
    exit 1
  fi
}

A100_ROOT=""
RTX5000_ROOT=""
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --a100-root)
      A100_ROOT="${2:-}"
      shift 2
      ;;
    --rtx5000-root)
      RTX5000_ROOT="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$A100_ROOT" ]]; then
  A100_ROOT="$PROJECT_ROOT/results/a100"
fi
if [[ -z "$RTX5000_ROOT" ]]; then
  RTX5000_ROOT="$PROJECT_ROOT/results/rtx5000"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$PROJECT_ROOT/results/compare"
fi
PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: python3/python not found" >&2
  exit 1
fi
require_python3 "$PYTHON_BIN"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_compare_envs.py" \
  --a100-root "$A100_ROOT" \
  --rtx5000-root "$RTX5000_ROOT" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_compare.py" \
  --compare-dir "$OUTPUT_DIR"

echo "Done: comparison artifacts generated in $OUTPUT_DIR"
