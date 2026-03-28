#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_campaign.sh --env a100|rtx5000 [options]

Options:
  --env NAME              Environment name: a100 or rtx5000 (required)
  --gpu ID                NVIDIA/nvidia-smi GPU index, resolved to UUID internally (default: 0)
  --sample-ms N           Power sampling period in ms, positive integer (default: 10)
  --energy-duration-ms N  Target in-process duration for each energy case in ms, positive integer (default: 2000)
  --stable-window-trim X  Fraction trimmed from start/end of run window, decimal in [0, 0.5) (default: 0.15)
  --skip-baseline         Skip baseline run (useful to resume energy/plots/validation)
  --skip-build            Skip cmake configure/build step
  --install-python-deps   Run pip install -r requirements.txt before execution
  --profile               Capture Nsight Systems BW/Compute/GEMM traces with timestamp
  --profile-only          Capture only Nsight Systems BW/Compute/GEMM traces and exit
  --help                  Show this help

Examples:
  ./scripts/run_campaign.sh --env a100
  ./scripts/run_campaign.sh --env rtx5000 --gpu 0 --energy-duration-ms 3000
  ./scripts/run_campaign.sh --env rtx5000 --gpu 0 --profile-only --skip-build
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not in PATH" >&2
    exit 1
  fi
}

require_python3() {
  local bin="$1"
  if ! "$bin" -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1; then
    echo "Error: '$bin' must resolve to a Python 3 interpreter" >&2
    exit 1
  fi
}

require_non_negative_int() {
  local value="$1"
  local name="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Error: $name must be a non-negative integer, got '$value'" >&2
    exit 1
  fi
}

require_positive_int() {
  local value="$1"
  local name="$2"
  require_non_negative_int "$value" "$name"
  if (( value <= 0 )); then
    echo "Error: $name must be > 0, got '$value'" >&2
    exit 1
  fi
}

require_trim_ratio() {
  local value="$1"
  local name="$2"
  if [[ ! "$value" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "Error: $name must be a decimal value in [0, 0.5), got '$value'" >&2
    exit 1
  fi

  "$PYTHON_BIN" - "$value" "$name" <<'PY'
import sys

value = float(sys.argv[1])
name = sys.argv[2]
if not (0.0 <= value < 0.5):
    raise SystemExit(f"Error: {name} must be in [0, 0.5), got '{sys.argv[1]}'")
PY
}

CURRENT_LOGGER_PID=""

cleanup_current_logger() {
  local logger_pid="${CURRENT_LOGGER_PID:-}"

  if [[ -z "$logger_pid" ]]; then
    return 0
  fi

  if kill -0 "$logger_pid" >/dev/null 2>&1; then
    kill "$logger_pid" >/dev/null 2>&1 || true
  fi

  wait "$logger_pid" 2>/dev/null || true
  CURRENT_LOGGER_PID=""
}

power_log_has_sample() {
  local log_path="$1"

  if [[ ! -s "$log_path" ]]; then
    return 1
  fi

  local last_line=""
  last_line="$(tail -n 1 "$log_path" 2>/dev/null || true)"
  [[ "$last_line" == *,*,*,*,*,* ]] || return 1
  [[ "$last_line" != *"N/A"* ]] || return 1
  [[ "$last_line" != *"n/a"* ]] || return 1
  [[ "$last_line" != *"Not Supported"* ]] || return 1
  [[ "$last_line" != *"not supported"* ]]
}

wait_for_power_logger_sample() {
  local log_path="$1"
  local logger_pid="$2"
  local label="$3"
  local max_attempts="${4:-100}"
  local attempt

  for ((attempt = 0; attempt < max_attempts; attempt++)); do
    if power_log_has_sample "$log_path"; then
      return 0
    fi

    if ! kill -0 "$logger_pid" >/dev/null 2>&1; then
      echo "Error: power logger exited before producing samples for case '$label'" >&2
      return 1
    fi

    sleep 0.1
  done

  echo "Error: timed out waiting for first power sample for case '$label' at $log_path" >&2
  return 1
}

handle_exit_signal() {
  local signal="$1"

  trap - EXIT INT TERM HUP
  cleanup_current_logger

  case "$signal" in
    INT)
      exit 130
      ;;
    TERM)
      exit 143
      ;;
    HUP)
      exit 129
      ;;
  esac
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

resolve_gpu_uuid() {
  local gpu_index="$1"
  local idx=""
  local uuid=""

  while IFS=, read -r idx uuid; do
    idx="$(echo "$idx" | xargs)"
    uuid="$(echo "$uuid" | xargs)"
    if [[ "$idx" == "$gpu_index" && -n "$uuid" ]]; then
      echo "$uuid"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)

  echo "Error: --gpu '$gpu_index' was not found in nvidia-smi GPU index list" >&2
  exit 1
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

load_case_field() {
  local perf_dir="$1"
  local case_key="$2"
  local field="$3"

  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/perf_targets.py" \
    --perf-dir "$perf_dir" \
    --case "$case_key" \
    --field "$field"
}

run_energy_case() {
  local label="$1"
  local case_key="$2"
  local bench_bin="$3"
  local out_root="$4"
  local gpu="$5"
  local sample_ms="$6"
  local energy_duration_ms="$7"
  local perf_dir="$8"

  local out_csv
  out_csv="$RESULTS_ROOT/energy/$(load_case_field "$perf_dir" "$case_key" power-log)"
  local out_meta
  out_meta="$RESULTS_ROOT/energy/$(load_case_field "$perf_dir" "$case_key" meta-log)"
  local -a bench_args
  read -r -a bench_args <<< "$(load_case_field "$perf_dir" "$case_key" args)"

  mkdir -p "$(dirname "$out_csv")" "$(dirname "$out_meta")"
  : > "$out_csv"

  cleanup_current_logger
  sed 's/\r$//' "${PROJECT_ROOT}/scripts/power_logger.sh" | bash -s -- "$gpu" "$out_csv" "$sample_ms" &
  CURRENT_LOGGER_PID=$!
  local logger_pid="$CURRENT_LOGGER_PID"

  if ! wait_for_power_logger_sample "$out_csv" "$logger_pid" "$label"; then
    cleanup_current_logger
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES="$gpu" "$bench_bin" \
    "${bench_args[@]}" \
    --energy-duration-ms "$energy_duration_ms" \
    --energy-meta-out "$out_meta" \
    --energy-case-key "$case_key" \
    --out-dir "$out_root/tmp" >/dev/null; then
    cleanup_current_logger
    echo "Error: long-run energy benchmark failed for case '$label'" >&2
    return 1
  fi

  # Short postroll so the logger can flush the last sample after the benchmark exits.
  sleep 0.5

  if ! kill -0 "$logger_pid" >/dev/null 2>&1; then
    echo "Warning: power logger exited early for case '$label'" >&2
  fi

  cleanup_current_logger

  if [[ ! -s "$out_csv" ]]; then
    echo "Error: empty power log for case '$label' at $out_csv" >&2
    return 1
  fi
  if [[ ! -s "$out_meta" ]]; then
    echo "Error: missing energy metadata for case '$label' at $out_meta" >&2
    return 1
  fi
}

run_nsys_profile() {
  local bench_bin="$1"
  local gpu="$2"
  local mode="$3"
  local out_prefix="$4"
  local tmp_nsys_dir="$5"
  local out_root="$6"

  mkdir -p "$(dirname "$out_prefix")" "$tmp_nsys_dir" "$out_root/tmp"

  TMPDIR="$tmp_nsys_dir" CUDA_VISIBLE_DEVICES="$gpu" \
    nsys profile --force-overwrite true --sample=none --cpuctxsw=none --trace=cuda,cublas \
      -o "$out_prefix" \
      "$bench_bin" --mode "$mode" --out-dir "$out_root/tmp"
}

run_nsys_profiles() {
  local bench_bin="$1"
  local gpu="$2"
  local results_root="$3"
  local tmp_nsys_dir="$4"
  local ts="$5"
  local step_label="$6"

  local mode
  for mode in bw compute gemm; do
    local out_prefix="$results_root/profiling/${mode}_trace_${ts}"
    echo "[$step_label] Capturing Nsight Systems ${mode} trace"
    if ! run_nsys_profile "$bench_bin" "$gpu" "$mode" "$out_prefix" "$tmp_nsys_dir" "$results_root"; then
      return 1
    fi
    echo "Done: profile trace at ${out_prefix}.nsys-rep"
  done
}

write_run_config() {
  local out_path="$1"
  local git_commit="unknown"
  local gpu_name=""
  local driver_version=""
  local cuda_driver_version=""
  local nvcc_version=""
  local python_version=""
  local cmd="./scripts/run_campaign.sh"

  if command -v git >/dev/null 2>&1; then
    git_commit="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  fi

  gpu_name="$(nvidia-smi -i "$GPU_UUID" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
  driver_version="$(nvidia-smi -i "$GPU_UUID" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
  cuda_driver_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1 | xargs || true)"
  nvcc_version="$("$NVCC_BIN" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -n1 | xargs || true)"
  python_version="$("$PYTHON_BIN" --version 2>&1 | xargs || true)"

  local arg
  for arg in "${ORIGINAL_ARGS[@]}"; do
    cmd+=" $(printf '%q' "$arg")"
  done

  cat > "$out_path" <<EOF
run_timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_timestamp_local=$(date +%Y-%m-%dT%H:%M:%S%z)
host=$(hostname)
user=$(whoami)
git_commit=$git_commit
project_root=$PROJECT_ROOT
environment=$ENV_NAME
gpu_index=$GPU
gpu_uuid=$GPU_UUID
gpu_name=${gpu_name:-unknown}
driver_version=${driver_version:-unknown}
cuda_driver_version=${cuda_driver_version:-unknown}
nvcc_version=${nvcc_version:-unknown}
nvcc_bin=$NVCC_BIN
python_bin=$PYTHON_BIN
python_version=${python_version:-unknown}
sample_ms=$SAMPLE_MS
energy_duration_ms=$ENERGY_DURATION_MS
stable_window_trim=$STABLE_WINDOW_TRIM
power_telemetry_source=nvidia-smi
power_scope=gpu_board
node_power_not_measured=1
activity_definition=benchmark_stable_window
skip_build=$SKIP_BUILD
skip_baseline=$SKIP_BASELINE
install_python_deps=$INSTALL_PY_DEPS
profile=$DO_PROFILE
profile_only=$PROFILE_ONLY
build_dir=$BUILD_DIR
results_root=$RESULTS_ROOT
bench_bin=$BENCH_BIN
command=$cmd
EOF
}

ENV_NAME=""
GPU="0"
SAMPLE_MS="10"
ENERGY_DURATION_MS="2000"
STABLE_WINDOW_TRIM="0.15"
SKIP_BASELINE="0"
SKIP_BUILD="0"
INSTALL_PY_DEPS="0"
DO_PROFILE="0"
PROFILE_ONLY="0"
ORIGINAL_ARGS=("$@")

trap cleanup_current_logger EXIT
trap 'handle_exit_signal INT' INT
trap 'handle_exit_signal TERM' TERM
trap 'handle_exit_signal HUP' HUP

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
    --energy-duration-ms)
      ENERGY_DURATION_MS="${2:-}"
      shift 2
      ;;
    --stable-window-trim)
      STABLE_WINDOW_TRIM="${2:-}"
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
NSYS_TMP_DIR="$RESULTS_ROOT/tmp/nsys_tmp"

PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: python3/python not found" >&2
  exit 1
fi
require_python3 "$PYTHON_BIN"

require_non_negative_int "$GPU" "--gpu"
require_positive_int "$SAMPLE_MS" "--sample-ms"
require_positive_int "$ENERGY_DURATION_MS" "--energy-duration-ms"
require_trim_ratio "$STABLE_WINDOW_TRIM" "--stable-window-trim"

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

GPU_UUID="$(resolve_gpu_uuid "$GPU")"
echo "Using GPU index $GPU -> UUID $GPU_UUID"

mkdir -p "$RESULTS_ROOT"/baseline "$RESULTS_ROOT"/energy "$RESULTS_ROOT"/profiling "$RESULTS_ROOT"/env "$RESULTS_ROOT"/tmp "$LOG_DIR"

if [[ "$INSTALL_PY_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

nvidia-smi > "$RESULTS_ROOT/env/env_nvidia_smi.txt"
"$NVCC_BIN" --version > "$RESULTS_ROOT/env/env_nvcc.txt"
"$PYTHON_BIN" --version > "$RESULTS_ROOT/env/env_python.txt" 2>&1
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

write_run_config "$RESULTS_ROOT/env/run_config.txt"
echo "Saved run config: $RESULTS_ROOT/env/run_config.txt"

if [[ "$PROFILE_ONLY" == "1" ]]; then
  require_cmd nsys
  TS="$(date +%Y%m%d_%H%M%S)"
  echo "[profile-only] Capturing Nsight Systems BW/Compute/GEMM traces"
  echo "[profile-only] Using TMPDIR=$NSYS_TMP_DIR"
  if ! run_nsys_profiles "$BENCH_BIN" "$GPU_UUID" "$RESULTS_ROOT" "$NSYS_TMP_DIR" "$TS" "profile-only"; then
    echo "Error: Nsight Systems profiling failed in profile-only mode." >&2
    echo "Hint: run 'nsys status --environment' and verify BENCH_BIN='$BENCH_BIN' is executable." >&2
    exit 1
  fi
  echo "Done: profile traces at $RESULTS_ROOT/profiling/{bw,compute,gemm}_trace_${TS}.nsys-rep"
  exit 0
fi

if [[ "$SKIP_BASELINE" != "1" ]]; then
  echo "[1/6] Running baseline benchmarks for $ENV_NAME"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$BENCH_BIN" --mode all --out-dir "$RESULTS_ROOT/baseline" | tee "$LOG_DIR/run_all_stdout.txt"
else
  echo "[1/6] Skipping baseline benchmarks (--skip-baseline)"
fi

PERF_DIR="$RESULTS_ROOT/baseline"

echo "[2/6] Logging bandwidth power cases"
run_energy_case "BW peak" "bw_peak" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"
run_energy_case "BW sustained" "bw_sustained" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"

echo "[3/6] Logging compute power cases"
run_energy_case "Compute FP32 peak" "compute_fp32_peak" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"
run_energy_case "Compute FP64 peak" "compute_fp64_peak" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"

echo "[4/6] Logging GEMM power cases"
run_energy_case "GEMM TF32=0 max" "gemm_tf32_0_max" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"
run_energy_case "GEMM TF32=1 max" "gemm_tf32_1_max" "$BENCH_BIN" "$RESULTS_ROOT" "$GPU_UUID" "$SAMPLE_MS" "$ENERGY_DURATION_MS" "$PERF_DIR"

echo "[5/6] Computing efficiency + plots"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/energy_active_summary.py" \
  --perf-dir "$RESULTS_ROOT/baseline" \
  --power-dir "$RESULTS_ROOT/energy" \
  --stable-window-trim "$STABLE_WINDOW_TRIM" \
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

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_fft.py" \
  --input "$RESULTS_ROOT/baseline/fft.csv" \
  --output "$RESULTS_ROOT/baseline/fft.png"

echo "[6/6] Validating outputs"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_results.py" \
  --results-dir "$RESULTS_ROOT/baseline" \
  --energy-summary "$RESULTS_ROOT/energy/efficiency_active_summary.csv"

if [[ "$DO_PROFILE" == "1" ]]; then
  require_cmd nsys
  TS="$(date +%Y%m%d_%H%M%S)"
  echo "[profile] Using TMPDIR=$NSYS_TMP_DIR"
  if ! run_nsys_profiles "$BENCH_BIN" "$GPU_UUID" "$RESULTS_ROOT" "$NSYS_TMP_DIR" "$TS" "profile"; then
    echo "Error: Nsight Systems profiling failed at end of campaign." >&2
    echo "Hint: run 'nsys status --environment' and verify BENCH_BIN='$BENCH_BIN' is executable." >&2
    exit 1
  fi
fi

echo "Done: $ENV_NAME campaign completed."
echo "Results at: $RESULTS_ROOT"
if [[ "$DO_PROFILE" != "1" ]]; then
  echo "Note: no Nsight trace generated. Re-run with --profile or --profile-only."
fi
