#include <cuda_runtime.h>

#include <cmath>
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
        x = fmaf(a, x, c);  // 2 FLOPs (mul+add)
        x = fmaf(b, x, c);  // +2 FLOPs
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

// Returns average seconds per launch.
template <typename LaunchFn>
static double time_kernel(LaunchFn launch, int warmup, int reps) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    for (int i = 0; i < warmup; i++) launch();
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup sync");

    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < reps; i++) launch();
    check(cudaGetLastError(), "timed launch");
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");

    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");

    return (ms / 1000.0) / (double)reps;
}

static double run_fp32_gflops(int grid, int block, int iters) {
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, sizeof(float)), "malloc fp32 out");
    check(cudaMemset(d_out, 0, sizeof(float)), "memset fp32 out");

    auto launch = [&]() { fma_fp32<<<grid, block>>>(d_out, iters); };

    double sec = time_kernel(launch, /*warmup=*/5, /*reps=*/50);
    check(cudaDeviceSynchronize(), "sync fp32");

    check(cudaFree(d_out), "free fp32 out");

    // 2 FMAs per iter, 2 FLOPs per FMA => 4 FLOPs/iter.
    double threads = (double)grid * (double)block;
    double flops = threads * (double)iters * 4.0;
    return (flops / sec) / 1e9;
}

static double run_fp64_gflops(int grid, int block, int iters) {
    double* d_out = nullptr;
    check(cudaMalloc(&d_out, sizeof(double)), "malloc fp64 out");
    check(cudaMemset(d_out, 0, sizeof(double)), "memset fp64 out");

    auto launch = [&]() { fma_fp64<<<grid, block>>>(d_out, iters); };

    double sec = time_kernel(launch, /*warmup=*/5, /*reps=*/50);
    check(cudaDeviceSynchronize(), "sync fp64");

    check(cudaFree(d_out), "free fp64 out");

    double threads = (double)grid * (double)block;
    double flops = threads * (double)iters * 4.0;
    return (flops / sec) / 1e9;
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

        double g32 = run_fp32_gflops(grid, block, iters);
        std::fprintf(f, "fp32,%d,%d,%d,%.2f\n", block, grid, iters, g32);
        std::fflush(f);
        std::printf("FP32 block=%d -> %.2f GFLOP/s\n", block, g32);

        double g64 = run_fp64_gflops(grid, block, iters);
        std::fprintf(f, "fp64,%d,%d,%d,%.2f\n", block, grid, iters, g64);
        std::fflush(f);
        std::printf("FP64 block=%d -> %.2f GFLOP/s\n", block, g64);
    }

    std::fclose(f);
}
