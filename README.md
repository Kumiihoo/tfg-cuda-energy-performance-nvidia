# TFG CUDA Benchmarks (A100 + RTX 5000)

CUDA/C++ benchmark suite for performance and energy-efficiency analysis across NVIDIA GPU generations.

## What is implemented

- CUDA benchmarks:
  - `bw`: global memory copy bandwidth
  - `compute`: FP32/FP64 FMA throughput
  - `gemm`: cuBLAS SGEMM with TF32 on/off
- Power telemetry via `nvidia-smi`
- Plot generation scripts
- Active-power efficiency summary
- CSV validator for reproducibility checks

## Project layout

- `src/`
  - `main.cu`: CLI entrypoint (`--mode`, `--out-dir`, `--tag`) with auto output routing by GPU type
  - `device_info.cu`: GPU inventory and checks
  - `bw_bench.cu`: BW sweep with VRAM-aware size cap
  - `compute_bench.cu`: compute sweep with SM-aware grid size
  - `gemm_bench.cu`: cuBLAS SGEMM benchmark
- `scripts/`
  - `power_logger.sh`: power logging helper
  - `energy_active_summary.py`: active-power efficiency summary
  - `plot_bw.py`, `plot_compute.py`, `plot_gemm.py`: plot generators
  - `validate_results.py`: CSV sanity validator
  - `run_campaign.sh`: one-command end-to-end automation by environment
- `results/`
  - `a100/`: A100 artifacts
  - `rtx5000/`: Quadro RTX 5000 artifacts

Current repository data has been reorganized so existing baseline/energy/profiling artifacts are under `results/a100/...`.

## Requirements

- Linux machine with NVIDIA GPU + driver
- CUDA Toolkit (validated with CUDA 11.8 toolchain)
- CMake >= 3.18
- C++ compiler
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

## Run benchmark suite manually

If you do not pass `--out-dir`, benchmark output is auto-routed to:

- `results/a100/baseline` on A100
- `results/rtx5000/baseline` on RTX 5000
- `results/other_gpu/baseline` otherwise

Examples:

```bash
# Auto output routing by detected GPU
CUDA_VISIBLE_DEVICES=0 ./build/a100/bench --mode all

# Explicit path override
CUDA_VISIBLE_DEVICES=0 ./build/rtx5000/bench --mode all --out-dir results/rtx5000/baseline
```

## One-command automation

Run full campaign for one environment (build, baseline, power logs, efficiency, plots, validation). Profiling is opt-in:

```bash
chmod +x scripts/run_campaign.sh
./scripts/run_campaign.sh --env a100 --gpu 0
./scripts/run_campaign.sh --env rtx5000 --gpu 0
```

Useful flags:

```bash
# Install Python deps automatically before run
./scripts/run_campaign.sh --env a100 --install-python-deps

# Skip build if already compiled in build/a100 or build/rtx5000
./scripts/run_campaign.sh --env a100 --skip-build

# Resume post-baseline stages only (power, summary, plots, validation)
./scripts/run_campaign.sh --env rtx5000 --skip-build --skip-baseline

# Capture Nsight Systems BW/Compute/GEMM traces with timestamp at end of campaign
./scripts/run_campaign.sh --env rtx5000 --profile

# Capture only Nsight BW/Compute/GEMM traces (no 1/6..6/6 pipeline)
./scripts/run_campaign.sh --env rtx5000 --profile-only --skip-build

# Profiling uses TMPDIR automatically at results/<env>/tmp/nsys_tmp
```

## Validation and plotting (manual)

```bash
python scripts/energy_active_summary.py \
  --perf-dir results/a100/baseline \
  --power-dir results/a100/energy \
  --output results/a100/energy/efficiency_active_summary.csv

python scripts/plot_bw.py --input results/a100/baseline/bw.csv --output results/a100/baseline/bw.png
python scripts/plot_compute.py --input results/a100/baseline/compute.csv --output results/a100/baseline/compute.png
python scripts/plot_gemm.py --input results/a100/baseline/gemm.csv --output results/a100/baseline/gemm.png

python scripts/validate_results.py --results-dir results/a100/baseline \
  --energy-summary results/a100/energy/efficiency_active_summary.csv
```

## Profiling

Use Nsight Systems only for timeline evidence, not for final throughput tables.

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

## Methodology notes

- Final performance values must come from non-profiled runs.
- BW traffic is reported as read + write for copy kernel.
- Compute grid size is derived from detected SM count.
- BW maximum size is capped automatically from available VRAM for portability.
- Active-power filtering uses an adaptive threshold based on observed SM clock.






