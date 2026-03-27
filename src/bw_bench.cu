#include <cuda_runtime.h>

#include <algorithm>
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

double run_bw_bytes_per_s(size_t bytes, int iters, int block) {
    size_t n = bytes / sizeof(float);
    if (n == 0) return -1.0;

    float* d_in = nullptr;
    float* d_out = nullptr;

    if (!try_alloc(&d_in, n * sizeof(float), "cudaMalloc d_in")) return -1.0;
    if (!try_alloc(&d_out, n * sizeof(float), "cudaMalloc d_out")) {
        check(cudaFree(d_in), "cudaFree d_in after d_out fail");
        return -1.0;
    }

    check(cudaMemset(d_in, 0, n * sizeof(float)), "cudaMemset d_in");
    check(cudaMemset(d_out, 0, n * sizeof(float)), "cudaMemset d_out");

    int grid = (int)((n + (size_t)block - 1) / (size_t)block);

    for (int i = 0; i < 5; i++) copy_kernel<<<grid, block>>>(d_in, d_out, n);
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    check(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iters; i++) copy_kernel<<<grid, block>>>(d_in, d_out, n);
    check(cudaGetLastError(), "timed launch");
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

    check(cudaEventDestroy(start), "event destroy start");
    check(cudaEventDestroy(stop), "event destroy stop");
    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_out), "cudaFree d_out");

    double seconds = ms / 1000.0;
    if (seconds <= 0.0) return -1.0;

    // Approximate DRAM traffic for copy kernel: read + write.
    double total_bytes = (double)bytes * (double)iters * 2.0;
    return total_bytes / seconds;
}

static void write_bw_row(FILE* f, size_t bytes, int iters, int block) {
    double bps = run_bw_bytes_per_s(bytes, iters, block);
    if (bps <= 0.0) {
        std::fprintf(stderr, "BW bytes=%zu failed or was skipped\n", bytes);
        std::exit(1);
    }

    double gbs = bps / 1e9;
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
