#include "bench_api.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void checkCufft(cufftResult status, const char* msg) {
    if (status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cuFFT error %s: status=%d\n", msg, (int)status);
        std::exit(1);
    }
}

class FftCaseRunner {
public:
    FftCaseRunner(int n, int batch, int iters) : n_(n), batch_(batch), iters_(iters) {
        checkCuda(cudaMalloc((void**)&data_, buffer_bytes()), "cudaMalloc fft buffer");
        checkCuda(cudaMemset(data_, 0, buffer_bytes()), "cudaMemset fft buffer");
        checkCufft(cufftPlan1d(&plan_, n_, CUFFT_C2C, batch_), "cufftPlan1d");
        checkCuda(cudaEventCreate(&start_), "fft event start");
        checkCuda(cudaEventCreate(&stop_), "fft event stop");
        warmup();
    }

    ~FftCaseRunner() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
        if (plan_) cufftDestroy(plan_);
        if (data_) cudaFree(data_);
    }

    void measure_case(double* time_ms, double* transforms_per_s, double* msamples_per_s) {
        const double measured_ms = run_case_repeats_ms(1);
        fill_metrics(measured_ms, 1, time_ms, transforms_per_s, msamples_per_s);
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
            total_samples(case_repeats) / (measured_ms / 1000.0) / 1e6,
            measured_ms,
            wall_ms,
            case_repeats,
            run_start,
            run_end,
        };
    }

private:
    void warmup() {
        for (int i = 0; i < 3; ++i) {
            launch_fft_once();
        }
        checkCuda(cudaDeviceSynchronize(), "fft warmup sync");
    }

    void launch_fft_once() {
        checkCufft(cufftExecC2C(plan_, data_, data_, CUFFT_FORWARD), "cufftExecC2C");
    }

    void launch_case_once() {
        for (int i = 0; i < iters_; ++i) {
            launch_fft_once();
        }
    }

    double estimate_case_ms() {
        return run_case_repeats_ms(1);
    }

    double run_case_repeats_ms(long long case_repeats) {
        checkCuda(cudaEventRecord(start_), "fft record start");
        for (long long repeat = 0; repeat < case_repeats; ++repeat) {
            launch_case_once();
        }
        checkCuda(cudaEventRecord(stop_), "fft record stop");
        checkCuda(cudaEventSynchronize(stop_), "fft sync stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "fft elapsed");
        return static_cast<double>(ms);
    }

    void fill_metrics(double measured_ms,
                      long long case_repeats,
                      double* time_ms,
                      double* transforms_per_s,
                      double* msamples_per_s) const {
        const double seconds = measured_ms / 1000.0;
        const double exec_calls = static_cast<double>(iters_) * static_cast<double>(case_repeats);
        const double transforms = static_cast<double>(batch_) * exec_calls;
        const double samples = static_cast<double>(n_) * transforms;

        if (seconds <= 0.0 || exec_calls <= 0.0) {
            *time_ms = -1.0;
            *transforms_per_s = -1.0;
            *msamples_per_s = -1.0;
            return;
        }

        *time_ms = measured_ms / exec_calls;
        *transforms_per_s = transforms / seconds;
        *msamples_per_s = samples / seconds / 1e6;
    }

    size_t buffer_bytes() const {
        return static_cast<size_t>(n_) * static_cast<size_t>(batch_) * sizeof(cufftComplex);
    }

    double total_samples(long long case_repeats) const {
        return static_cast<double>(n_) * static_cast<double>(batch_) * static_cast<double>(iters_) *
               static_cast<double>(case_repeats);
    }

    int n_ = 0;
    int batch_ = 0;
    int iters_ = 0;
    cufftHandle plan_ = 0;
    cufftComplex* data_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

void run_fft_case_metrics(int n, int batch, int iters, double* time_ms, double* transforms_per_s, double* msamples_per_s) {
    FftCaseRunner runner(n, batch, iters);
    runner.measure_case(time_ms, transforms_per_s, msamples_per_s);
}

EnergyRunResult run_fft_case_for_energy(int n, int batch, int iters, double target_duration_ms) {
    FftCaseRunner runner(n, batch, iters);
    return runner.run_for_energy(target_duration_ms);
}

static void write_fft_row(FILE* f, int n, int batch, int iters) {
    double time_ms = 0.0;
    double transforms_per_s = 0.0;
    double msamples_per_s = 0.0;
    run_fft_case_metrics(n, batch, iters, &time_ms, &transforms_per_s, &msamples_per_s);

    std::fprintf(f, "%d,%d,%d,%.4f,%.2f,%.2f\n", n, batch, iters, time_ms, transforms_per_s, msamples_per_s);
    std::fflush(f);
    std::printf("FFT C2C N=%d batch=%d -> %.2f MSamples/s\n", n, batch, msamples_per_s);
}

void run_fft_sweep_to_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "n,batch,iters,time_ms,transforms_per_s,MSamples_per_s\n");

    cudaDeviceProp p{};
    checkCuda(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");

    const size_t target_bytes = std::max<size_t>(
        32ull << 20, std::min<size_t>(128ull << 20, static_cast<size_t>(p.totalGlobalMem * 0.10)));
    const int iters = 10;
    const std::vector<int> sizes = {1 << 14, 1 << 16, 1 << 18, 786432, 1 << 20};

    std::printf("FFT sweep target buffer size capped near %zu bytes (%.2f MiB)\n",
                target_bytes, static_cast<double>(target_bytes) / (1024.0 * 1024.0));

    for (int n : sizes) {
        const size_t bytes_per_transform = static_cast<size_t>(n) * sizeof(cufftComplex);
        const int batch = static_cast<int>(std::max<size_t>(
            1, std::min<size_t>(2048, target_bytes / std::max<size_t>(bytes_per_transform, 1))));
        write_fft_row(f, n, batch, iters);
    }

    std::fclose(f);
}

void run_fft_case_to_csv(const char* path, int n, int batch, int iters) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "n,batch,iters,time_ms,transforms_per_s,MSamples_per_s\n");
    write_fft_row(f, n, batch, iters);
    std::fclose(f);
}
