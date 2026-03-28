# TFG CUDA Benchmarks (A100 + RTX 5000)

CUDA/C++ benchmark suite for performance and energy-efficiency analysis across NVIDIA GPU generations.

## What is implemented

- CUDA benchmarks:
  - `bw`: global memory copy bandwidth
  - `compute`: FP32/FP64 FMA throughput
  - `gemm`: cuBLAS SGEMM with TF32 on/off
  - `fft`: cuFFT 1D batched C2C control workload
- Power telemetry via `nvidia-smi`
- Plot generation scripts (per-GPU and cross-GPU)
- Stable-window efficiency summary with per-target power logs and benchmark-delimited metadata
- CSV validators for reproducibility checks
- End-to-end automation by environment

## Project layout

- `src/`
  - `main.cu`: CLI entrypoint (`--mode`, `--out-dir`, `--tag`, `--energy-duration-ms`, `--energy-meta-out`, optional `--energy-case-key`) with case-specific flags for BW/Compute/GEMM/FFT
  - `bench_api.h`: shared benchmark/energy-run interface used by the CUDA benchmark implementations
  - `device_info.cu`: GPU inventory and checks
  - `bw_bench.cu`: BW sweep with VRAM-aware size cap
  - `compute_bench.cu`: compute sweep with SM-aware grid size
  - `gemm_bench.cu`: cuBLAS SGEMM benchmark with in-process long-run energy mode
  - `fft_bench.cu`: cuFFT 1D batched C2C benchmark
- `scripts/`
  - `power_logger.sh`: power logging helper
  - `perf_targets.py`: selects the energy target cases from baseline CSVs
  - `energy_active_summary.py`: benchmark-delimited stable-window efficiency summary
  - `plot_bw.py`, `plot_compute.py`, `plot_gemm.py`, `plot_fft.py`: per-benchmark plot generators
  - `plot_compare_envs.py`: A100 vs RTX5000 comparison CSV + plots
  - `validate_results.py`: single-environment CSV sanity validator
  - `validate_compare.py`: cross-environment artifact validator
  - `run_campaign.sh`: one-command end-to-end automation by environment
  - `run_compare.sh`: one-command cross-environment comparison + validation
  - `archive/`: legacy utilities kept only for historical reference, not for current methodology
- `results/`
  - `a100/`: A100 artifacts
  - `rtx5000/`: Quadro RTX 5000 artifacts
  - `compare/`: cross-environment comparison artifacts

## Requirements

- Linux machine with NVIDIA GPU + driver
- CUDA Toolkit (validated with CUDA 11.8+ toolchains)
- CMake >= 3.18
- `make`
- C++ compiler (`g++`, as required by the current automation scripts)
- `nvidia-smi` (driver CLI; required)
- `nsys` (Nsight Systems CLI; required only when using `--profile` / `--profile-only`)
- Python 3 with packages in `requirements.txt`
- `git` optional: used only to stamp `git_commit` in `run_config.txt` when available

Install Python deps:

```bash
PYTHON_BIN=python
command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1 || PYTHON_BIN=python3
"$PYTHON_BIN" -m pip install -r requirements.txt
```

## Build

Use separate build folders per environment:

```bash
cmake -S . -B build/a100 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/a100 -j

cmake -S . -B build/rtx5000 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/rtx5000 -j
```

## One-command automation (per environment)

Run full campaign for one environment (build, baseline, long-run power logs, efficiency, plots, validation). Profiling is opt-in:

```bash
chmod +x scripts/run_campaign.sh
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader

# Pick the NVIDIA GPU index that matches the board you want to benchmark.
export GPU_INDEX=0  # replace 0 with the NVIDIA index reported by nvidia-smi

./scripts/run_campaign.sh --env a100 --gpu "$GPU_INDEX" --profile
./scripts/run_campaign.sh --env rtx5000 --gpu "$GPU_INDEX" --profile
```

Useful flags:

```bash
# Install Python deps automatically before run
./scripts/run_campaign.sh --env a100 --install-python-deps

# Skip build if already compiled in build/a100 or build/rtx5000
./scripts/run_campaign.sh --env a100 --skip-build

# Increase long-run duration for energy measurements
./scripts/run_campaign.sh --env rtx5000 --energy-duration-ms 3000

# Tighten or relax the stable-window trim used by the energy summary
./scripts/run_campaign.sh --env rtx5000 --stable-window-trim 0.10

# Resume post-baseline stages only (energy, summary, plots, validation)
./scripts/run_campaign.sh --env rtx5000 --skip-build --skip-baseline

# Capture Nsight Systems BW/Compute/GEMM/FFT traces with timestamp at end of campaign
./scripts/run_campaign.sh --env rtx5000 --profile

# Capture only Nsight BW/Compute/GEMM/FFT traces (no 1/7..7/7 pipeline)
./scripts/run_campaign.sh --env rtx5000 --profile-only --skip-build
```

Runner argument constraints:

- `--gpu` must be a non-negative NVIDIA `nvidia-smi` GPU index; the runner resolves it to the matching GPU UUID so `nvidia-smi` logging and CUDA execution target the same physical board.
- `--sample-ms` and `--energy-duration-ms` must be positive integers.
- `--stable-window-trim` must stay in `[0, 0.5)`.
- For case-specific `bench` runs, numeric case parameters must be `> 0`; `--bw-bytes` must also be a multiple of `sizeof(float)`.

Each campaign stores execution metadata in:

```text
results/<env>/env/run_config.txt
```

This file is part of the reproducibility artifacts (host, GPU, driver/CUDA versions, sampling period, long-run duration, trim ratio, flags, command line, paths, and `git_commit` when available).

## Cross-environment comparison (A100 vs RTX5000)

After both environments have completed campaigns:

```bash
chmod +x scripts/run_compare.sh
./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
```

`run_compare.sh` regenerates `--output-dir` from scratch, so stale files from older comparisons do not survive into the new artifact set.

If campaigns were executed on different servers, copy one environment results tree to the server where you run the comparison. In a fresh clone, the repo no longer tracks an empty `results/rtx5000/` tree, so the following command creates the expected path directly. If `results/rtx5000` already exists from an older copy, remove it first to avoid nested `results/rtx5000/rtx5000` paths.

```bash
rm -rf ~/tfg/results/rtx5000  # only needed if that local folder already exists from a previous copy
scp -r user_rtx@10.222.1.134:~/tfg/results/rtx5000 ~/tfg/results/rtx5000  # replace user_rtx with the remote username
./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
```

This generates:

- `results/compare/summary_compare.csv`
- `results/compare/environment_compare.csv`
- `results/compare/methodology_notes.txt`
- `results/compare/perf_absolute_compare.png`
- `results/compare/efficiency_compare.png`
- `results/compare/speedup_a100_vs_rtx5000.png`

The comparison summary reports both `BW peak` and `BW sustained`, and now includes FFT performance and efficiency when the baseline and energy artifacts are available in both environments.
Performance and efficiency plots separate incompatible units into different subplots, so `GB/s` is not mixed with `GFLOP/s` or `MSamples/s`.
`environment_compare.csv` and `methodology_notes.txt` make stack mismatches and measurement-scope limitations explicit.
`run_compare.sh` now allows cross-server software stack differences such as driver/CUDA driver versions, `nvcc`, or Python version to remain as warnings in the comparison metadata so the compare artifacts can still be generated for demos and heterogeneous lab servers.
`run_compare.sh` still fails validation when blocking measurement-methodology fields differ across environments, namely sampling period, energy duration, trim ratio, or the documented telemetry/scope metadata.
`git_commit` is exported as informational reproducibility metadata when `git` is available, but missing `git` no longer blocks cross-environment comparison.

## Preflight (environment checks only)

Run this checklist on each server before final runs:

```bash
# 1) Enter project
cd ~/tfg

# 2) Activate virtual environment (venv or .venv)
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# 3) Select the Python 3 interpreter that will also be used by the automation scripts
export PYTHON_BIN=python
command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1 || export PYTHON_BIN=python3

# 4) Install Python deps into that same interpreter
"$PYTHON_BIN" -m pip install -r requirements.txt

# 5) Normalize line endings and script permissions
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/run_campaign.sh scripts/run_compare.sh scripts/power_logger.sh

# 6) Check shell and CUDA compiler paths
which bash
which nvcc

# 7) Check build/runtime tools required by run_campaign.sh
which cmake
which make
which g++

# 8) Check Nsight Systems path (needed only for --profile / --profile-only)
which nsys || echo "nsys not found: profiling will be unavailable"

# 9) Check driver/GPU visibility and note the GPU index you will pass to --gpu
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
nvidia-smi | head -n 5

# 10) Verify the automation scripts parse correctly after line-ending normalization
./scripts/run_campaign.sh --help >/dev/null
./scripts/run_compare.sh --help >/dev/null
```

For reproducible A100 vs RTX5000 comparisons, keep `--sample-ms`, `--energy-duration-ms`, and `--stable-window-trim` identical across servers, and keep the telemetry/scope metadata aligned. Driver/CUDA/`nvcc`/Python drift is still reported in the comparison metadata, but it no longer blocks `run_compare.sh` on heterogeneous servers. If `git` is available on both servers, running both campaigns from the same repo commit is still strongly recommended for traceability, but it is no longer required to make `run_compare.sh` succeed.

## Two-server execution (A100 + RTX5000)

Typical sequence:

```bash
# On A100 server
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
export A100_GPU_INDEX=0  # replace 0 with the A100 NVIDIA index reported by nvidia-smi
./scripts/run_campaign.sh --env a100 --gpu "$A100_GPU_INDEX" --sample-ms 10 --energy-duration-ms 2000 --stable-window-trim 0.15 --profile

# On RTX5000 server
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
export RTX5000_GPU_INDEX=0  # replace 0 with the RTX5000 NVIDIA index reported by nvidia-smi
./scripts/run_campaign.sh --env rtx5000 --gpu "$RTX5000_GPU_INDEX" --sample-ms 10 --energy-duration-ms 2000 --stable-window-trim 0.15 --profile

# Back on A100 server: copy RTX5000 results and compare
rm -rf ~/tfg/results/rtx5000  # only needed if that local folder already exists from a previous copy
scp -r user_rtx@10.222.1.134:~/tfg/results/rtx5000 ~/tfg/results/rtx5000  # replace user_rtx with the remote username
./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
```

## Validation commands (manual)

Reuse the same `PYTHON_BIN` selected in the preflight step, or re-run the interpreter selection first:

```bash
export PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)" >/dev/null 2>&1 || export PYTHON_BIN=python3
```

Single environment:

```bash
"$PYTHON_BIN" scripts/validate_results.py \
  --results-dir results/a100/baseline \
  --energy-summary results/a100/energy/efficiency_active_summary.csv
```

Cross environment:

```bash
"$PYTHON_BIN" scripts/validate_compare.py --compare-dir results/compare
```

## Profiling

Use Nsight Systems for timeline evidence, not for final throughput tables.

```bash
# Select the NVIDIA GPU index first if not already exported in this shell
export GPU_INDEX="${GPU_INDEX:-0}"  # replace 0 with the NVIDIA index reported by nvidia-smi

# Automatic profiling (bw + compute + gemm + fft) at end of campaign
./scripts/run_campaign.sh --env a100 --gpu "$GPU_INDEX" --profile

# Profiling only (bw + compute + gemm + fft)
./scripts/run_campaign.sh --env a100 --gpu "$GPU_INDEX" --profile-only --skip-build

# Expected artifacts
ls -lh results/a100/profiling/bw_trace_*.nsys-rep \
       results/a100/profiling/compute_trace_*.nsys-rep \
       results/a100/profiling/gemm_trace_*.nsys-rep \
       results/a100/profiling/fft_trace_*.nsys-rep
```

## Final packaging (archive all artifacts)

Create a timestamped archive including both environments and comparison outputs:

```bash
tar -czf "results/tfg_results_$(date +%Y%m%d_%H%M%S).tgz" results/a100 results/rtx5000 results/compare
```

## Methodology notes

- Final performance values must come from non-profiled runs.
- BW traffic is reported as read + write for copy kernel.
- BW summaries distinguish `BW peak` (max observed GB/s) from `BW sustained` (last, largest-size sweep point).
- Power logs are captured per selected target case (`BW peak`, `BW sustained`, `Compute FP32 peak`, `Compute FP64 peak`, `GEMM TF32=0 max`, `GEMM TF32=1 max`, `FFT C2C max`).
- Energy runs execute each selected case in-process until the requested duration is reached, instead of stitching together many short process invocations.
- Each energy case writes a metadata CSV with measured work time, wall time, case repeats, UTC timestamps, and source-host local wall-clock timestamps for the benchmark interval.
- Campaign runs pass an explicit energy `case_key` into the metadata so each power log can be matched back to the selected baseline target deterministically.
- `run_campaign.sh` validates its public numeric CLI arguments early: `--gpu >= 0`, `--sample-ms > 0`, `--energy-duration-ms > 0`, and `--stable-window-trim in [0, 0.5)`.
- `run_campaign.sh` resolves `--gpu` from an NVIDIA driver index to the matching GPU UUID before launching the logger, CUDA benchmark, or Nsight profiling, so all three stages address the same physical GPU.
- Power telemetry comes from `nvidia-smi` and reflects GPU-board power only; this project does not measure whole-node/system power.
- `git_commit` is recorded in `run_config.txt` when `git` is available, but it is treated as informational metadata rather than a strict comparison blocker.
- Cross-environment comparison now exports metadata warnings so driver/toolchain or sampling mismatches are visible in `results/compare/environment_compare.csv` and `results/compare/methodology_notes.txt`.
- Driver/CUDA/`nvcc`/Python mismatches remain non-blocking warnings in the comparison metadata; blocking validation is reserved for measurement-methodology mismatches such as sample period, energy duration, trim ratio, telemetry source, or power scope.
- Compute grid size is derived from detected SM count.
- BW maximum size is capped automatically from available VRAM for portability.
- FFT baseline uses 1D batched C2C transforms with throughput reported in `MSamples/s`.
- FFT is integrated as a baseline/control benchmark, an automated energy-summary target, and a cross-environment performance/efficiency comparison case.
- Efficiency uses the benchmark-delimited stable window: first crop the logger to the benchmark start/end timestamps with strict overlap, then trim the configured fraction from the beginning and end of that run window.
- Stable-window recropping uses the local benchmark timestamps recorded in the metadata, so summaries can be recomputed on another machine without depending on that machine's current timezone.
- In `efficiency_active_summary.csv`, `Perf` now refers to the long-run execution used for power; baseline performance is retained as an audit column so power and efficiency come from the same run.
- SM clock is retained as a diagnostic signal, not as the primary definition of activity.
