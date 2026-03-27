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
- C++ compiler (`g++` or compatible)
- `nvidia-smi` (driver CLI; required)
- `nsys` (Nsight Systems CLI; required only when using `--profile` / `--profile-only`)
- Python 3 with packages in `requirements.txt`

Install Python deps:

```bash
python -m pip install -r requirements.txt
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
./scripts/run_campaign.sh --env a100 --gpu 0 --profile
./scripts/run_campaign.sh --env rtx5000 --gpu 0 --profile
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

# Capture Nsight Systems BW/Compute/GEMM traces with timestamp at end of campaign
./scripts/run_campaign.sh --env rtx5000 --profile

# Capture only Nsight BW/Compute/GEMM traces (no 1/6..6/6 pipeline)
./scripts/run_campaign.sh --env rtx5000 --profile-only --skip-build
```

Each campaign stores execution metadata in:

```text
results/<env>/env/run_config.txt
```

This file is part of the reproducibility artifacts (host, GPU, driver/CUDA versions, sampling period, long-run duration, trim ratio, flags, command line, paths).

## Cross-environment comparison (A100 vs RTX5000)

After both environments have completed campaigns:

```bash
chmod +x scripts/run_compare.sh
./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
```

If campaigns were executed on different servers, copy one environment results tree to the server where you run the comparison, for example:

```bash
scp -r <user_rtx>@10.222.1.134:~/tfg/results/rtx5000 ~/tfg/results/rtx5000
```

This generates:

- `results/compare/summary_compare.csv`
- `results/compare/environment_compare.csv`
- `results/compare/methodology_notes.txt`
- `results/compare/perf_absolute_compare.png`
- `results/compare/efficiency_compare.png`
- `results/compare/speedup_a100_vs_rtx5000.png`

The comparison summary reports both `BW peak` and `BW sustained`, and now includes FFT performance when `fft.csv` is available in both environments.
FFT currently contributes to the performance comparison only; the efficiency comparison still covers the original energy-tracked cases.
Performance and efficiency plots separate incompatible units into different subplots, so `GB/s` is not mixed with `GFLOP/s` or `MSamples/s`.
`environment_compare.csv` and `methodology_notes.txt` make stack mismatches and measurement-scope limitations explicit.

## Preflight (environment checks only)

Run this checklist on each server before final runs:

```bash
# 1) Enter project
cd ~/tfg

# 2) Activate virtual environment (venv or .venv)
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# 3) Install Python deps
python -m pip install -r requirements.txt

# 4) Normalize line endings and script permissions
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/run_campaign.sh scripts/run_compare.sh scripts/power_logger.sh

# 5) Check CUDA compiler path
which nvcc

# 6) Check Nsight Systems path (needed for profiling)
which nsys

# 7) Check driver/GPU visibility
nvidia-smi | head -n 5
```

## Two-server execution (A100 + RTX5000)

Typical sequence:

```bash
# On A100 server
./scripts/run_campaign.sh --env a100 --gpu 0 --profile

# On RTX5000 server
./scripts/run_campaign.sh --env rtx5000 --gpu 0 --profile

# Back on A100 server: copy RTX5000 results and compare
scp -r <user_rtx>@10.222.1.134:~/tfg/results/rtx5000 ~/tfg/results/rtx5000
./scripts/run_compare.sh --a100-root results/a100 --rtx5000-root results/rtx5000 --output-dir results/compare
```

## Validation commands (manual)

Single environment:

```bash
python scripts/validate_results.py \
  --results-dir results/a100/baseline \
  --energy-summary results/a100/energy/efficiency_active_summary.csv
```

Cross environment:

```bash
python scripts/validate_compare.py --compare-dir results/compare
```

## Profiling

Use Nsight Systems for timeline evidence, not for final throughput tables.

```bash
# Automatic profiling (bw + compute + gemm) at end of campaign
./scripts/run_campaign.sh --env a100 --gpu 0 --profile

# Profiling only (bw + compute + gemm)
./scripts/run_campaign.sh --env a100 --gpu 0 --profile-only --skip-build

# Expected artifacts
ls -lh results/a100/profiling/bw_trace_*.nsys-rep \
       results/a100/profiling/compute_trace_*.nsys-rep \
       results/a100/profiling/gemm_trace_*.nsys-rep
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
- Power logs are captured per selected target case (`BW peak`, `BW sustained`, `Compute FP32 peak`, `Compute FP64 peak`, `GEMM TF32=0 max`, `GEMM TF32=1 max`).
- Energy runs execute each selected case in-process until the requested duration is reached, instead of stitching together many short process invocations.
- Each energy case writes a metadata CSV with measured work time, wall time, case repeats, and benchmark start/end timestamps.
- Campaign runs pass an explicit energy `case_key` into the metadata so each power log can be matched back to the selected baseline target deterministically.
- Power telemetry comes from `nvidia-smi` and reflects GPU-board power only; this project does not measure whole-node/system power.
- Cross-environment comparison now exports metadata warnings so driver/toolchain or sampling mismatches are visible in `results/compare/environment_compare.csv` and `results/compare/methodology_notes.txt`.
- Compute grid size is derived from detected SM count.
- BW maximum size is capped automatically from available VRAM for portability.
- FFT baseline uses 1D batched C2C transforms with throughput reported in `MSamples/s`.
- FFT is currently integrated as a baseline/control benchmark plus cross-environment performance comparison; it is not yet part of the automated energy summary target set.
- Efficiency uses the benchmark-delimited stable window: first crop the logger to the benchmark start/end timestamps with strict overlap, then trim the configured fraction from the beginning and end of that run window.
- In `efficiency_active_summary.csv`, `Perf` now refers to the long-run execution used for power; baseline performance is retained as an audit column so power and efficiency come from the same run.
- SM clock is retained as a diagnostic signal, not as the primary definition of activity.
