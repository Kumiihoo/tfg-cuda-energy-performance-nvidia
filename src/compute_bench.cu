#include "bench_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static int detect_sm_count() {
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");
    return p.multiProcessorCount > 0 ? p.multiProcessorCount : 1;
}

// Kernel FP32: many FMAs per thread. A final store prevents dead-code elimination.
__global__ void fma_fp32(float* out, int iters) {
    float a = 1.000001f, b = 1.0000007f, c = 0.999999f;
    float x = 0.12345f;

#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        x = fmaf(a, x, c);
        x = fmaf(b, x, c);
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) out[0] = x;
}

// Kernel FP64
__global__ void fma_fp64(double* out, int iters) {
    double a = 1.0000000000001, b = 1.0000000000007, c = 0.9999999999999;
    double x = 0.123456789;

#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        x = fma(a, x, c);
        x = fma(b, x, c);
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) out[0] = x;
}

class ComputeCaseRunner {
public:
    ComputeCaseRunner(const char* dtype, int block, int grid, int iters)
        : block_(block), grid_(grid), iters_(iters) {
        fp64_ = std::strcmp(dtype, "fp64") == 0;
        if (!fp64_ && std::strcmp(dtype, "fp32") != 0) {
            std::fprintf(stderr, "Unsupported compute dtype '%s'\n", dtype);
            std::exit(1);
        }

        if (fp64_) {
            check(cudaMalloc((void**)&d_out_fp64_, sizeof(double)), "malloc fp64 out");
            check(cudaMemset(d_out_fp64_, 0, sizeof(double)), "memset fp64 out");
        } else {
            check(cudaMalloc((void**)&d_out_fp32_, sizeof(float)), "malloc fp32 out");
            check(cudaMemset(d_out_fp32_, 0, sizeof(float)), "memset fp32 out");
        }

        check(cudaEventCreate(&start_), "event create start");
        check(cudaEventCreate(&stop_), "event create stop");

        warmup();
    }

    ~ComputeCaseRunner() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
        if (d_out_fp32_) cudaFree(d_out_fp32_);
        if (d_out_fp64_) cudaFree(d_out_fp64_);
    }

    double measure_case_gflops() {
        return flops_to_gflops(case_flops(), run_case_repeats_ms(1));
    }

    EnergyRunResult run_for_energy(double target_duration_ms) {
        const double estimate_ms = std::max(estimate_case_ms(), 0.001);
        long long case_repeats = std::max<long long>(
            1, static_cast<long long>(std::ceil((target_duration_ms / estimate_ms) * 1.10)));

        double measured_ms = run_case_repeats_ms(case_repeats);
        while (measured_ms + 1e-6 < target_duration_ms) {
            const double batch_ms = std::max(measured_ms / static_cast<double>(case_repeats), 0.001);
            const long long extra_repeats = std::max<long long>(
                1, static_cast<long long>(std::ceil(((target_duration_ms - measured_ms) / batch_ms) * 1.10)));
            measured_ms += run_case_repeats_ms(extra_repeats);
            case_repeats += extra_repeats;
        }

        return EnergyRunResult{
            flops_to_gflops(case_flops() * static_cast<double>(case_repeats), measured_ms),
            measured_ms,
            case_repeats,
        };
    }

private:
    static constexpr int kWarmupLaunches = 5;
    static constexpr int kCaseKernelLaunches = 50;

    void warmup() {
        for (int i = 0; i < kWarmupLaunches; ++i) {
            launch_kernel_once();
        }
        check(cudaGetLastError(), "warmup launch");
        check(cudaDeviceSynchronize(), "warmup sync");
    }

    void launch_kernel_once() {
        if (fp64_) {
            fma_fp64<<<grid_, block_>>>(d_out_fp64_, iters_);
        } else {
            fma_fp32<<<grid_, block_>>>(d_out_fp32_, iters_);
        }
    }

    void launch_case_once() {
        for (int i = 0; i < kCaseKernelLaunches; ++i) {
            launch_kernel_once();
        }
    }

    double estimate_case_ms() {
        return run_case_repeats_ms(1);
    }

    double run_case_repeats_ms(long long case_repeats) {
        check(cudaEventRecord(start_), "record start");
        for (long long repeat = 0; repeat < case_repeats; ++repeat) {
            launch_case_once();
        }
        check(cudaGetLastError(), "timed launch");
        check(cudaEventRecord(stop_), "record stop");
        check(cudaEventSynchronize(stop_), "sync stop");

        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_, stop_), "elapsed");
        return static_cast<double>(ms);
    }

    double case_flops() const {
        const double threads = static_cast<double>(grid_) * static_cast<double>(block_);
        const double flops_per_kernel = threads * static_cast<double>(iters_) * 4.0;
        return flops_per_kernel * static_cast<double>(kCaseKernelLaunches);
    }

    static double flops_to_gflops(double total_flops, double ms) {
        const double seconds = ms / 1000.0;
        if (seconds <= 0.0) return -1.0;
        return (total_flops / seconds) / 1e9;
    }

    int block_ = 0;
    int grid_ = 0;
    int iters_ = 0;
    bool fp64_ = false;
    float* d_out_fp32_ = nullptr;
    double* d_out_fp64_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

double run_compute_case_gflops(const char* dtype, int block, int grid, int iters) {
    ComputeCaseRunner runner(dtype, block, grid, iters);
    return runner.measure_case_gflops();
}

EnergyRunResult run_compute_case_for_energy(const char* dtype, int block, int grid, int iters, double target_duration_ms) {
    ComputeCaseRunner runner(dtype, block, grid, iters);
    return runner.run_for_energy(target_duration_ms);
}

static void write_compute_row(FILE* f, const char* dtype, int block, int grid, int iters) {
    const double gflops = run_compute_case_gflops(dtype, block, grid, iters);

    std::fprintf(f, "%s,%d,%d,%d,%.2f\n", dtype, block, grid, iters, gflops);
    std::fflush(f);
    std::printf("%s block=%d grid=%d -> %.2f GFLOP/s\n", dtype, block, grid, gflops);
}

void run_compute_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "dtype,block,grid,iters,GFLOPs\n");

    const int sms = detect_sm_count();
    const int base_grid = sms * 8;
    const int iters = 4096;

    int blocks[] = {128, 256, 512, 1024};

    std::printf("Compute sweep uses %d SMs, grid=%d\n", sms, base_grid);

    for (int block : blocks) {
        int grid = base_grid;
        write_compute_row(f, "fp32", block, grid, iters);
        write_compute_row(f, "fp64", block, grid, iters);
    }

    std::fclose(f);
}

void run_compute_case_to_csv(const char* path, const char* dtype, int block, int grid, int iters) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "dtype,block,grid,iters,GFLOPs\n");
    write_compute_row(f, dtype, block, grid, iters);
    std::fclose(f);
}
