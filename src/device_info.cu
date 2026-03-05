#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

void print_device_info() {
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    std::printf("Devices visible to this process: %d\n", n);

    if (n <= 0) {
        std::fprintf(stderr, "No CUDA devices visible.\n");
        std::exit(1);
    }

    const int dev = 0;
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

    std::printf("Using device %d: %s\n", dev, p.name);
    std::printf("  Compute capability: %d.%d\n", p.major, p.minor);
    std::printf("  SMs: %d\n", p.multiProcessorCount);
    std::printf("  Global mem: %.1f GB\n", (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("  Warp size: %d\n", p.warpSize);
    std::printf("  Max threads/block: %d\n", p.maxThreadsPerBlock);
}
