#include "bench_api.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
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

class GemmCaseRunner {
public:
    GemmCaseRunner(int n, int iters, bool tf32) : n_(n), iters_(iters), tf32_(tf32) {
        checkCublas(cublasCreate(&handle_), "cublasCreate");
        checkCublas(cublasSetMathMode(handle_, tf32_ ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH),
                    "cublasSetMathMode");

        const size_t bytes = matrix_bytes();
        const size_t elems = matrix_elems();
        checkCuda(cudaMalloc((void**)&a_, bytes), "malloc A");
        checkCuda(cudaMalloc((void**)&b_, bytes), "malloc B");
        checkCuda(cudaMalloc((void**)&c_, bytes), "malloc C");

        const int block = 256;
        const int grid = (int)((elems + (size_t)block - 1) / (size_t)block);
        fill_kernel<<<grid, block>>>(a_, elems, 1.0f);
        fill_kernel<<<grid, block>>>(b_, elems, 1.0f);
        checkCuda(cudaGetLastError(), "fill kernel");
        checkCuda(cudaMemset(c_, 0, bytes), "memset C");
        checkCuda(cudaDeviceSynchronize(), "sync fill");

        checkCuda(cudaEventCreate(&start_), "event start");
        checkCuda(cudaEventCreate(&stop_), "event stop");

        warmup();
    }

    ~GemmCaseRunner() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
        if (a_) cudaFree(a_);
        if (b_) cudaFree(b_);
        if (c_) cudaFree(c_);
        if (handle_) cublasDestroy(handle_);
    }

    double measure_case_gflops() {
        return ops_to_gflops(case_ops(), run_case_repeats_ms(1));
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
            ops_to_gflops(case_ops() * static_cast<double>(case_repeats), measured_ms),
            measured_ms,
            case_repeats,
        };
    }

private:
    void warmup() {
        for (int i = 0; i < 5; ++i) {
            launch_gemm_once();
        }
        checkCuda(cudaDeviceSynchronize(), "sync warmup");
    }

    void launch_gemm_once() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                n_, n_, n_,
                                &alpha,
                                a_, n_,
                                b_, n_,
                                &beta,
                                c_, n_), "cublasSgemm");
    }

    void launch_case_once() {
        for (int i = 0; i < iters_; ++i) {
            launch_gemm_once();
        }
    }

    double estimate_case_ms() {
        return run_case_repeats_ms(1);
    }

    double run_case_repeats_ms(long long case_repeats) {
        checkCuda(cudaEventRecord(start_), "record start");
        for (long long repeat = 0; repeat < case_repeats; ++repeat) {
            launch_case_once();
        }
        checkCuda(cudaEventRecord(stop_), "record stop");
        checkCuda(cudaEventSynchronize(stop_), "sync stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "elapsed");
        return static_cast<double>(ms);
    }

    size_t matrix_elems() const {
        return (size_t)n_ * (size_t)n_;
    }

    size_t matrix_bytes() const {
        return matrix_elems() * sizeof(float);
    }

    double case_ops() const {
        return 2.0 * static_cast<double>(n_) * static_cast<double>(n_) * static_cast<double>(n_) *
               static_cast<double>(iters_);
    }

    static double ops_to_gflops(double total_ops, double ms) {
        const double seconds = ms / 1000.0;
        if (seconds <= 0.0) return -1.0;
        return (total_ops / seconds) / 1e9;
    }

    int n_ = 0;
    int iters_ = 0;
    bool tf32_ = false;
    cublasHandle_t handle_ = nullptr;
    float* a_ = nullptr;
    float* b_ = nullptr;
    float* c_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

double run_gemm_case_gflops(int n, int iters, bool tf32) {
    GemmCaseRunner runner(n, iters, tf32);
    return runner.measure_case_gflops();
}

EnergyRunResult run_gemm_case_for_energy(int n, int iters, bool tf32, double target_duration_ms) {
    GemmCaseRunner runner(n, iters, tf32);
    return runner.run_for_energy(target_duration_ms);
}

static void write_gemm_row(FILE* f, int n, int iters, bool tf32) {
    const double gflops = run_gemm_case_gflops(n, iters, tf32);
    std::fprintf(f, "%d,%d,%d,%.2f\n", n, iters, tf32 ? 1 : 0, gflops);
    std::fflush(f);
    std::printf("GEMM FP32 N=%d TF32=%d -> %.2f GFLOP/s\n", n, tf32 ? 1 : 0, gflops);
}

void run_gemm_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "N,iters,tf32,GFLOPs\n");

    int sizes[] = {1024, 2048, 4096, 8192};
    const int iters = 20;

    for (int n : sizes) {
        write_gemm_row(f, n, iters, false);
        write_gemm_row(f, n, iters, true);
    }

    std::fclose(f);
}

void run_gemm_case_to_csv(const char* path, int n, int iters, bool tf32) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "N,iters,tf32,GFLOPs\n");
    write_gemm_row(f, n, iters, tf32);
    std::fclose(f);
}
