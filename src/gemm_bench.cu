#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

__global__ void fill_kernel(float* data, size_t n, float value) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error %s: status=%d\n", msg, (int)s);
        std::exit(1);
    }
}

// GFLOPs para GEMM: 2*N^3 / time
static double gflops_gemm(int N, double seconds) {
    double ops = 2.0 * (double)N * (double)N * (double)N;
    return (ops / seconds) / 1e9;
}

double run_gemm_fp32(int N, int iters, bool tf32) {
    cublasHandle_t h;
    checkCublas(cublasCreate(&h), "cublasCreate");

    // TF32 on/off (solo afecta a FP32)
    checkCublas(cublasSetMathMode(h, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode");

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    size_t elems = (size_t)N * (size_t)N;
    float *A=nullptr, *B=nullptr, *C=nullptr;
    checkCuda(cudaMalloc(&A, bytes), "malloc A");
    checkCuda(cudaMalloc(&B, bytes), "malloc B");
    checkCuda(cudaMalloc(&C, bytes), "malloc C");
    int block = 256;
    int grid = (int)((elems + (size_t)block - 1) / (size_t)block);
    fill_kernel<<<grid, block>>>(A, elems, 1.0f);
    fill_kernel<<<grid, block>>>(B, elems, 1.0f);
    checkCuda(cudaGetLastError(), "fill kernel");
    checkCuda(cudaMemset(C, 0, bytes), "memset C");
    checkCuda(cudaDeviceSynchronize(), "sync fill");

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i=0;i<5;i++) {
        checkCublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                A, N,
                                B, N,
                                &beta,
                                C, N), "cublasSgemm warmup");
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event start");
    checkCuda(cudaEventCreate(&stop), "event stop");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i=0;i<iters;i++) {
        checkCublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                A, N,
                                B, N,
                                &beta,
                                C, N), "cublasSgemm");
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");

    float ms=0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    double sec = (ms/1000.0) / (double)iters;

    checkCuda(cudaEventDestroy(start), "destroy start");
    checkCuda(cudaEventDestroy(stop), "destroy stop");
    checkCuda(cudaFree(A), "free A");
    checkCuda(cudaFree(B), "free B");
    checkCuda(cudaFree(C), "free C");
    checkCublas(cublasDestroy(h), "cublasDestroy");

    return gflops_gemm(N, sec);
}

static void write_gemm_row(FILE* f, int N, int iters, bool tf32) {
    double gflops = run_gemm_fp32(N, iters, tf32);
    std::fprintf(f, "%d,%d,%d,%.2f\n", N, iters, tf32 ? 1 : 0, gflops);
    std::fflush(f);
    std::printf("GEMM FP32 N=%d TF32=%d -> %.2f GFLOP/s\n", N, tf32 ? 1 : 0, gflops);
}

void run_gemm_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) { perror("fopen"); std::exit(1); }

    std::fprintf(f, "N,iters,tf32,GFLOPs\n");

    int sizes[] = {1024, 2048, 4096, 8192};
    const int iters = 20;

    for (int N : sizes) {
        write_gemm_row(f, N, iters, false);
        write_gemm_row(f, N, iters, true);
    }

    std::fclose(f);
}

void run_gemm_case_to_csv(const char* path, int N, int iters, bool tf32) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "N,iters,tf32,GFLOPs\n");
    write_gemm_row(f, N, iters, tf32);
    std::fclose(f);
}
