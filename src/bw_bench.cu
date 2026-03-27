#include "bench_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static bool try_alloc(float** ptr, size_t bytes, const char* tag) {
    cudaError_t e = cudaMalloc((void**)ptr, bytes);
    if (e == cudaSuccess) return true;

    if (e == cudaErrorMemoryAllocation) {
        std::fprintf(stderr, "Skipping allocation for %s (%zu bytes): out of memory\n", tag, bytes);
        cudaGetLastError();
        return false;
    }

    check(e, tag);
    return false;
}

class BwCaseRunner {
public:
    BwCaseRunner(size_t bytes, int iters, int block) : bytes_(bytes), iters_(iters), block_(block) {
        n_ = bytes_ / sizeof(float);
        if (n_ == 0) {
            std::fprintf(stderr, "BW case requires at least one float element\n");
            std::exit(1);
        }

        if (!try_alloc(&d_in_, n_ * sizeof(float), "cudaMalloc d_in")) {
            std::exit(1);
        }
        if (!try_alloc(&d_out_, n_ * sizeof(float), "cudaMalloc d_out")) {
            check(cudaFree(d_in_), "cudaFree d_in after d_out fail");
            std::exit(1);
        }

        check(cudaMemset(d_in_, 0, n_ * sizeof(float)), "cudaMemset d_in");
        check(cudaMemset(d_out_, 0, n_ * sizeof(float)), "cudaMemset d_out");

        grid_ = (int)((n_ + (size_t)block_ - 1) / (size_t)block_);

        check(cudaEventCreate(&start_), "event create start");
        check(cudaEventCreate(&stop_), "event create stop");

        warmup();
    }

    ~BwCaseRunner() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
        if (d_in_) cudaFree(d_in_);
        if (d_out_) cudaFree(d_out_);
    }

    double measure_case_gbs() {
        return bytes_to_gbs(case_bytes_traffic(), run_case_repeats_ms(1));
    }

    EnergyRunResult run_for_energy(double target_duration_ms) {
        const double estimate_ms = std::max(estimate_case_ms(), 0.001);
        long long case_repeats = std::max<long long>(
            1, static_cast<long long>(std::ceil((target_duration_ms / estimate_ms) * 1.10)));

        const auto run_start = std::chrono::system_clock::now();
        double measured_ms = run_case_repeats_ms(case_repeats);
        while (measured_ms + 1e-6 < target_duration_ms) {
            const double batch_ms = std::max(measured_ms / static_cast<double>(case_repeats), 0.001);
            const long long extra_repeats = std::max<long long>(
                1, static_cast<long long>(std::ceil(((target_duration_ms - measured_ms) / batch_ms) * 1.10)));
            measured_ms += run_case_repeats_ms(extra_repeats);
            case_repeats += extra_repeats;
        }
        const auto run_end = std::chrono::system_clock::now();
        const double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(run_end - run_start).count();

        return EnergyRunResult{
            bytes_to_gbs(case_bytes_traffic() * static_cast<double>(case_repeats), measured_ms),
            measured_ms,
            wall_ms,
            case_repeats,
            run_start,
            run_end,
        };
    }

private:
    void warmup() {
        launch_case_once();
        launch_case_once();
        check(cudaGetLastError(), "warmup launch");
        check(cudaDeviceSynchronize(), "warmup sync");
    }

    void launch_case_once() {
        for (int i = 0; i < iters_; ++i) {
            copy_kernel<<<grid_, block_>>>(d_in_, d_out_, n_);
        }
    }

    double estimate_case_ms() {
        return run_case_repeats_ms(1);
    }

    double run_case_repeats_ms(long long case_repeats) {
        check(cudaEventRecord(start_), "event record start");
        for (long long repeat = 0; repeat < case_repeats; ++repeat) {
            launch_case_once();
        }
        check(cudaGetLastError(), "timed launch");
        check(cudaEventRecord(stop_), "event record stop");
        check(cudaEventSynchronize(stop_), "event sync stop");

        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_, stop_), "elapsed time");
        return static_cast<double>(ms);
    }

    double case_bytes_traffic() const {
        return static_cast<double>(bytes_) * static_cast<double>(iters_) * 2.0;
    }

    static double bytes_to_gbs(double total_bytes, double ms) {
        const double seconds = ms / 1000.0;
        if (seconds <= 0.0) return -1.0;
        return (total_bytes / seconds) / 1e9;
    }

    size_t bytes_ = 0;
    int iters_ = 0;
    int block_ = 0;
    size_t n_ = 0;
    int grid_ = 0;
    float* d_in_ = nullptr;
    float* d_out_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

double run_bw_case_gbs(size_t bytes, int iters, int block) {
    BwCaseRunner runner(bytes, iters, block);
    return runner.measure_case_gbs();
}

EnergyRunResult run_bw_case_for_energy(size_t bytes, int iters, int block, double target_duration_ms) {
    BwCaseRunner runner(bytes, iters, block);
    return runner.run_for_energy(target_duration_ms);
}

static void write_bw_row(FILE* f, size_t bytes, int iters, int block) {
    double gbs = run_bw_case_gbs(bytes, iters, block);
    if (gbs <= 0.0) {
        std::fprintf(stderr, "BW bytes=%zu failed or was skipped\n", bytes);
        std::exit(1);
    }

    std::fprintf(f, "%zu,%d,%d,%.3f\n", bytes, iters, block, gbs);
    std::fflush(f);
    std::printf("BW bytes=%zu -> %.3f GB/s\n", bytes, gbs);
}

void run_bw_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "bytes,iters,block,GBs\n");

    const int block = 256;
    const int iters = 200;

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");

    // Keep each vector at <=30% of VRAM so two allocations fit across GPUs.
    size_t max_test_bytes = (size_t)((double)p.totalGlobalMem * 0.30);
    max_test_bytes = std::max<size_t>(max_test_bytes, 1ull << 20);

    std::vector<size_t> candidates = {
        1ull << 20,
        4ull << 20,
        16ull << 20,
        64ull << 20,
        256ull << 20,
        1ull << 30,
        2ull << 30,
        4ull << 30,
        8ull << 30,
    };

    std::vector<size_t> sizes;
    for (size_t bytes : candidates) {
        if (bytes <= max_test_bytes) sizes.push_back(bytes);
    }

    if (sizes.empty()) sizes.push_back(1ull << 20);

    std::printf("BW sweep max size capped at %zu bytes (%.2f GB)\n",
                max_test_bytes, (double)max_test_bytes / (1024.0 * 1024.0 * 1024.0));

    for (size_t bytes : sizes) {
        write_bw_row(f, bytes, iters, block);
    }

    std::fclose(f);
}

void run_bw_case_to_csv(const char* path, size_t bytes, int iters, int block) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "bytes,iters,block,GBs\n");
    write_bw_row(f, bytes, iters, block);
    std::fclose(f);
}
