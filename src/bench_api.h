#pragma once

#include <chrono>
#include <cstddef>

struct EnergyRunResult {
    double avg_perf = 0.0;
    double measured_work_ms = 0.0;
    double wall_ms = 0.0;
    long long case_repeats = 0;
    std::chrono::system_clock::time_point run_start{};
    std::chrono::system_clock::time_point run_end{};
};

void print_device_info();

void run_bw_sweep_to_csv(const char* path);
void run_bw_case_to_csv(const char* path, size_t bytes, int iters, int block);
double run_bw_case_gbs(size_t bytes, int iters, int block);
EnergyRunResult run_bw_case_for_energy(size_t bytes, int iters, int block, double target_duration_ms);

void run_compute_sweep_to_csv(const char* path);
void run_compute_case_to_csv(const char* path, const char* dtype, int block, int grid, int iters);
double run_compute_case_gflops(const char* dtype, int block, int grid, int iters);
EnergyRunResult run_compute_case_for_energy(const char* dtype, int block, int grid, int iters, double target_duration_ms);

void run_gemm_sweep_to_csv(const char* path);
void run_gemm_case_to_csv(const char* path, int N, int iters, bool tf32);
double run_gemm_case_gflops(int N, int iters, bool tf32);
EnergyRunResult run_gemm_case_for_energy(int N, int iters, bool tf32, double target_duration_ms);

void run_fft_sweep_to_csv(const char* path);
void run_fft_case_to_csv(const char* path, int n, int batch, int iters);
void run_fft_case_metrics(int n, int batch, int iters, double* time_ms, double* transforms_per_s, double* msamples_per_s);
EnergyRunResult run_fft_case_for_energy(int n, int batch, int iters, double target_duration_ms);
