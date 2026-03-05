#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

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
    float *A=nullptr, *B=nullptr, *C=nullptr;
    checkCuda(cudaMalloc(&A, bytes), "malloc A");
    checkCuda(cudaMalloc(&B, bytes), "malloc B");
    checkCuda(cudaMalloc(&C, bytes), "malloc C");
    checkCuda(cudaMemset(A, 1, bytes), "memset A");
    checkCuda(cudaMemset(B, 1, bytes), "memset B");
    checkCuda(cudaMemset(C, 0, bytes), "memset C");

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

void run_gemm_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) { perror("fopen"); std::exit(1); }

    std::fprintf(f, "N,iters,tf32,GFLOPs\n");

    int sizes[] = {1024, 2048, 4096, 8192};
    const int iters = 20;

    for (int N : sizes) {
        double g_def = run_gemm_fp32(N, iters, false);
        std::fprintf(f, "%d,%d,%d,%.2f\n", N, iters, 0, g_def);
        std::fflush(f);
        std::printf("GEMM FP32 N=%d TF32=0 -> %.2f GFLOP/s\n", N, g_def);

        double g_tf = run_gemm_fp32(N, iters, true);
        std::fprintf(f, "%d,%d,%d,%.2f\n", N, iters, 1, g_tf);
        std::fflush(f);
        std::printf("GEMM FP32 N=%d TF32=1 -> %.2f GFLOP/s\n", N, g_tf);
    }

    std::fclose(f);
}
