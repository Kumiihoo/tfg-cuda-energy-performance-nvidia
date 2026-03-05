#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_campaign.sh --env a100|rtx5000 [options]

Options:
  --env NAME              Environment name: a100 or rtx5000 (required)
  --gpu ID                CUDA visible GPU index (default: 0)
  --sample-ms N           Power sampling period in ms (default: 10)
  --gemm-repeats N        GEMM repetitions for power run (default: 30)
  --compute-repeats N     Compute repetitions for power run (default: 50)
  --bw-repeats N          BW repetitions for power run (default: 10)
  --skip-baseline         Skip baseline run (useful to resume power/plots/validation)
  --skip-build            Skip cmake configure/build step
  --install-python-deps   Run pip install -r requirements.txt before execution
  --profile               Capture one Nsight Systems GEMM trace with timestamp
  --profile-only          Capture only Nsight Systems trace and exit
  --help                  Show this help

Examples:
  ./scripts/run_campaign.sh --env a100
  ./scripts/run_campaign.sh --env rtx5000 --gpu 0 --profile
  ./scripts/run_campaign.sh --env rtx5000 --gpu 0 --profile-only --skip-build
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not in PATH" >&2
    exit 1
  fi
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  echo ""
}

find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return
  fi

  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    echo "/usr/local/cuda/bin/nvcc"
    return
  fi

  local cand
  cand="$(ls -1 /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "$cand" && -x "$cand" ]]; then
    echo "$cand"
    return
  fi

  echo ""
}

run_power_loop() {
  local mode="$1"
  local repeats="$2"
  local out_csv="$3"
  local bench_bin="$4"
  local out_root="$5"
  local gpu="$6"
  local sample_ms="$7"

  mkdir -p "$(dirname "$out_csv")"
  : > "$out_csv"

  local attempt
  local local_repeats="$repeats"
  for attempt in 1 2; do
    sed 's/\r$//' "${PROJECT_ROOT}/scripts/power_logger.sh" | bash -s -- "$gpu" "$out_csv" "$sample_ms" &
    local lp=$!

    # Let the logger initialize so very short runs do not race startup.
    sleep 1.0

    for ((i = 1; i <= local_repeats; i++)); do
      CUDA_VISIBLE_DEVICES="$gpu" "$bench_bin" --mode "$mode" --out-dir "$out_root/tmp" >/dev/null
    done

    # Give logger a short grace period to flush first samples.
    local waited=0
    while [[ ! -s "$out_csv" && $waited -lt 30 ]]; do
      sleep 0.1
      waited=$((waited + 1))
    done

    if ! kill -0 "$lp" >/dev/null 2>&1; then
      echo "Warning: power logger exited early for mode '$mode' (attempt ${attempt})" >&2
    fi

    kill "$lp" >/dev/null 2>&1 || true
    wait "$lp" 2>/dev/null || true

    if [[ -s "$out_csv" ]]; then
      return 0
    fi

    echo "Warning: empty power log for mode '$mode' (attempt ${attempt}); retrying with more repeats..." >&2
    local_repeats=$((local_repeats * 2))
  done

  echo "Error: failed to capture power log for mode '$mode' at $out_csv" >&2
  return 1
}

ENV_NAME=""
GPU="0"
SAMPLE_MS="10"
GEMM_REPEATS="30"
COMPUTE_REPEATS="50"
BW_REPEATS="10"
SKIP_BASELINE="0"
SKIP_BUILD="0"
INSTALL_PY_DEPS="0"
DO_PROFILE="0"
PROFILE_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --gpu)
      GPU="${2:-}"
      shift 2
      ;;
    --sample-ms)
      SAMPLE_MS="${2:-}"
      shift 2
      ;;
    --gemm-repeats)
      GEMM_REPEATS="${2:-}"
      shift 2
      ;;
    --compute-repeats)
      COMPUTE_REPEATS="${2:-}"
      shift 2
      ;;
    --bw-repeats)
      BW_REPEATS="${2:-}"
      shift 2
      ;;
    --skip-baseline)
      SKIP_BASELINE="1"
      shift
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift
      ;;
    --install-python-deps)
      INSTALL_PY_DEPS="1"
      shift
      ;;
    --profile-only)
      DO_PROFILE="1"
      PROFILE_ONLY="1"
      shift
      ;;
    --profile)
      DO_PROFILE="1"
      shift
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

if [[ -z "$ENV_NAME" ]]; then
  echo "Error: --env is required" >&2
  usage
  exit 1
fi

case "$ENV_NAME" in
  a100)
    CUDA_ARCH="80"
    ;;
  rtx5000)
    CUDA_ARCH="75"
    ;;
  *)
    echo "Error: --env must be 'a100' or 'rtx5000'" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results/$ENV_NAME"
BUILD_ROOT="$PROJECT_ROOT/build"
BUILD_DIR="$BUILD_ROOT/$ENV_NAME"
LEGACY_BUILD_DIR="$PROJECT_ROOT/build_$ENV_NAME"
BENCH_BIN="$BUILD_DIR/bench"
LOG_DIR="$RESULTS_ROOT/logs"

PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: python3/python not found" >&2
  exit 1
fi

NVCC_BIN="$(find_nvcc)"
if [[ -z "$NVCC_BIN" ]]; then
  echo "Error: nvcc not found in PATH or /usr/local/cuda*/bin" >&2
  exit 1
fi

echo "Using nvcc: $NVCC_BIN"

# Force stable C locale to avoid CSV decimal-comma issues across servers.
export LC_ALL=C
export LANG=C

require_cmd cmake
require_cmd make
require_cmd nvidia-smi
require_cmd "$PYTHON_BIN"
require_cmd g++

mkdir -p "$RESULTS_ROOT"/baseline "$RESULTS_ROOT"/energy "$RESULTS_ROOT"/profiling "$RESULTS_ROOT"/env "$RESULTS_ROOT"/tmp "$LOG_DIR"

if [[ "$INSTALL_PY_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

nvidia-smi > "$RESULTS_ROOT/env/env_nvidia_smi.txt"
"$NVCC_BIN" --version > "$RESULTS_ROOT/env/env_nvcc.txt"
g++ --version > "$RESULTS_ROOT/env/env_gpp.txt"
cmake --version > "$RESULTS_ROOT/env/env_cmake.txt"

if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
    -DCMAKE_MAKE_PROGRAM="$(command -v make)"

  cmake --build "$BUILD_DIR" -j"$(nproc)"
fi

if [[ ! -x "$BENCH_BIN" && -x "$LEGACY_BUILD_DIR/bench" ]]; then
  echo "Warning: using legacy build dir $LEGACY_BUILD_DIR (rebuild to migrate into $BUILD_DIR)." >&2
  BENCH_BIN="$LEGACY_BUILD_DIR/bench"
fi

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Error: bench binary not found at $BUILD_DIR/bench" >&2
  echo "Run without --skip-build or check cmake output." >&2
  exit 1
fi

if [[ "$PROFILE_ONLY" == "1" ]]; then
  require_cmd nsys
  TS="$(date +%Y%m%d_%H%M%S)"
  echo "[profile-only] Capturing Nsight Systems GEMM trace"
  CUDA_VISIBLE_DEVICES="$GPU" nsys profile --force-overwrite true \
    -o "$RESULTS_ROOT/profiling/gemm_trace_${TS}" \
    "$BENCH_BIN" --mode gemm --out-dir "$RESULTS_ROOT/tmp"
  echo "Done: profile trace at $RESULTS_ROOT/profiling/gemm_trace_${TS}.nsys-rep"
  exit 0
fi

if [[ "$SKIP_BASELINE" != "1" ]]; then
  echo "[1/6] Running baseline benchmarks for $ENV_NAME"
  CUDA_VISIBLE_DEVICES="$GPU" "$BENCH_BIN" --mode all --out-dir "$RESULTS_ROOT/baseline" | tee "$LOG_DIR/run_all_stdout.txt"
else
  echo "[1/6] Skipping baseline benchmarks (--skip-baseline)"
fi

echo "[2/6] Logging GEMM power"
run_power_loop gemm "$GEMM_REPEATS" "$RESULTS_ROOT/energy/power_gemm_long.csv" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU" "$SAMPLE_MS"

echo "[3/6] Logging compute power"
run_power_loop compute "$COMPUTE_REPEATS" "$RESULTS_ROOT/energy/power_compute_long.csv" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU" "$SAMPLE_MS"

echo "[4/6] Logging bandwidth power"
run_power_loop bw "$BW_REPEATS" "$RESULTS_ROOT/energy/power_bw_long.csv" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU" "$SAMPLE_MS"

echo "[5/6] Computing efficiency + plots"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/energy_active_summary.py" \
  --perf-dir "$RESULTS_ROOT/baseline" \
  --power-dir "$RESULTS_ROOT/energy" \
  --output "$RESULTS_ROOT/energy/efficiency_active_summary.csv"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_bw.py" \
  --input "$RESULTS_ROOT/baseline/bw.csv" \
  --output "$RESULTS_ROOT/baseline/bw.png"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_compute.py" \
  --input "$RESULTS_ROOT/baseline/compute.csv" \
  --output "$RESULTS_ROOT/baseline/compute.png"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_gemm.py" \
  --input "$RESULTS_ROOT/baseline/gemm.csv" \
  --output "$RESULTS_ROOT/baseline/gemm.png"

echo "[6/6] Validating outputs"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_results.py" \
  --results-dir "$RESULTS_ROOT/baseline" \
  --energy-summary "$RESULTS_ROOT/energy/efficiency_active_summary.csv"

if [[ "$DO_PROFILE" == "1" ]]; then
  require_cmd nsys
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES="$GPU" nsys profile --force-overwrite true \
    -o "$RESULTS_ROOT/profiling/gemm_trace_${TS}" \
    "$BENCH_BIN" --mode gemm --out-dir "$RESULTS_ROOT/tmp"
fi

echo "Done: $ENV_NAME campaign completed."
echo "Results at: $RESULTS_ROOT"
if [[ "$DO_PROFILE" != "1" ]]; then
  echo "Note: no Nsight trace generated. Re-run with --profile or --profile-only."
fi






