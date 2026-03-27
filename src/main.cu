#include "bench_api.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>

static bool valid_mode(const std::string& mode) {
    return mode == "all" || mode == "bw" || mode == "compute" || mode == "gemm" || mode == "fft";
}

static bool valid_compute_dtype(const std::string& dtype) {
    return dtype == "fp32" || dtype == "fp64";
}

static bool parse_int_arg(const std::string& value, int* out) {
    try {
        size_t pos = 0;
        long long parsed = std::stoll(value, &pos, 10);
        if (pos != value.size()) return false;
        if (parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
        *out = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_size_arg(const std::string& value, size_t* out) {
    try {
        size_t pos = 0;
        unsigned long long parsed = std::stoull(value, &pos, 10);
        if (pos != value.size()) return false;
        *out = static_cast<size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

static std::string parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    return path.substr(0, slash);
}

static bool ensure_dir_recursive(const std::string& dir, std::string* err) {
    if (dir.empty()) return true;

    std::string current;
    if (!dir.empty() && dir[0] == '/') current = "/";

    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;

        if (!current.empty() && current.back() != '/') current += '/';
        current += part;

        struct stat st{};
        if (stat(current.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
                if (err) *err = "path exists but is not a directory: " + current;
                return false;
            }
            continue;
        }

        if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            if (err) *err = std::string("mkdir failed for '") + current + "': " + std::strerror(errno);
            return false;
        }
    }

    return true;
}

static std::string detect_env_folder() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return "unknown_gpu";

    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown_gpu";

    const std::string name = p.name;
    if (name.find("A100") != std::string::npos) return "a100";
    if (name.find("RTX 5000") != std::string::npos) return "rtx5000";
    return "other_gpu";
}

static std::string csv_escape(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += '"';
        out += c;
    }
    out += "\"";
    return out;
}

static std::string format_utc_timestamp(std::chrono::system_clock::time_point tp) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    const std::time_t tt = system_clock::to_time_t(tp);

    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif

    char base[32];
    std::strftime(base, sizeof(base), "%Y-%m-%dT%H:%M:%S", &tm);

    char out[40];
    std::snprintf(out, sizeof(out), "%s.%03lldZ", base, static_cast<long long>(ms.count()));
    return out;
}

static void write_bw_result_csv(const std::string& path, size_t bytes, int iters, int block, double gbs) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }
    std::fprintf(f, "bytes,iters,block,GBs\n");
    std::fprintf(f, "%zu,%d,%d,%.3f\n", bytes, iters, block, gbs);
    std::fclose(f);
}

static void write_compute_result_csv(const std::string& path, const std::string& dtype, int block, int grid, int iters,
                                     double gflops) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }
    std::fprintf(f, "dtype,block,grid,iters,GFLOPs\n");
    std::fprintf(f, "%s,%d,%d,%d,%.2f\n", dtype.c_str(), block, grid, iters, gflops);
    std::fclose(f);
}

static void write_gemm_result_csv(const std::string& path, int n, int iters, bool tf32, double gflops) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }
    std::fprintf(f, "N,iters,tf32,GFLOPs\n");
    std::fprintf(f, "%d,%d,%d,%.2f\n", n, iters, tf32 ? 1 : 0, gflops);
    std::fclose(f);
}

static void write_fft_result_csv(const std::string& path,
                                 int n,
                                 int batch,
                                 int iters,
                                 double time_ms,
                                 double transforms_per_s,
                                 double msamples_per_s) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }
    std::fprintf(f, "n,batch,iters,time_ms,transforms_per_s,MSamples_per_s\n");
    std::fprintf(f, "%d,%d,%d,%.4f,%.2f,%.2f\n", n, batch, iters, time_ms, transforms_per_s, msamples_per_s);
    std::fclose(f);
}

static void write_energy_meta_csv(const std::string& path,
                                  const std::string& case_key,
                                  const std::string& params,
                                  const std::string& perf_unit,
                                  double avg_perf,
                                  double target_duration_ms,
                                  double measured_work_ms,
                                  double wall_ms,
                                  long long case_repeats,
                                  const std::string& run_start_utc,
                                  const std::string& run_end_utc) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        perror("fopen");
        std::exit(1);
    }

    std::fprintf(f,
                 "case_key,params,perf_unit,avg_perf,target_duration_ms,measured_work_ms,wall_ms,case_repeats,run_start_utc,run_end_utc\n");
    std::fprintf(f, "%s,%s,%s,%.6f,%.3f,%.3f,%.3f,%lld,%s,%s\n",
                 csv_escape(case_key).c_str(),
                 csv_escape(params).c_str(),
                 csv_escape(perf_unit).c_str(),
                 avg_perf,
                 target_duration_ms,
                 measured_work_ms,
                 wall_ms,
                 case_repeats,
                 csv_escape(run_start_utc).c_str(),
                 csv_escape(run_end_utc).c_str());
    std::fclose(f);
}

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [--mode all|bw|compute|gemm|fft] [--out-dir PATH] [--tag NAME] [advanced case flags]\n"
        "Default out-dir: auto -> results/<a100|rtx5000|other_gpu>/baseline\n"
        "Energy mode:\n"
        "  --energy-duration-ms N   Target duration for in-process long-run measurement (case mode only)\n"
        "  --energy-meta-out PATH   Metadata CSV written for energy mode (required with --energy-duration-ms)\n"
        "Advanced case flags:\n"
        "  BW:      --bw-bytes N --bw-iters N --bw-block N\n"
        "  Compute: --compute-dtype fp32|fp64 --compute-block N --compute-grid N --compute-iters N\n"
        "  GEMM:    --gemm-n N --gemm-iters N --gemm-tf32 0|1\n"
        "  FFT:     --fft-n N --fft-batch N --fft-iters N\n"
        "Examples:\n"
        "  %s --mode all\n"
        "  %s --mode gemm --out-dir results/a100/baseline\n"
        "  %s --mode compute --compute-dtype fp32 --compute-block 1024 --compute-grid 864 --compute-iters 4096\n"
        "  %s --mode fft --fft-n 1048576 --fft-batch 16 --fft-iters 10\n"
        "  %s --mode bw --bw-bytes 1048576 --bw-iters 200 --bw-block 256 --energy-duration-ms 2000 \\\n"
        "     --energy-meta-out results/a100/energy/power_bw_peak_long_meta.csv\n",
        prog, prog, prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string mode = "all";
    std::string out_dir;
    std::string tag;
    std::string energy_meta_out;
    bool out_dir_given = false;
    size_t bw_bytes = 0;
    int bw_iters = 0;
    int bw_block = 0;
    int energy_duration_ms = 0;
    bool bw_bytes_given = false;
    bool bw_iters_given = false;
    bool bw_block_given = false;
    bool energy_duration_given = false;
    bool energy_meta_out_given = false;
    std::string compute_dtype;
    int compute_block = 0;
    int compute_grid = 0;
    int compute_iters = 0;
    bool compute_dtype_given = false;
    bool compute_block_given = false;
    bool compute_grid_given = false;
    bool compute_iters_given = false;
    int gemm_n = 0;
    int gemm_iters = 0;
    int gemm_tf32 = -1;
    int fft_n = 0;
    int fft_batch = 0;
    int fft_iters = 0;
    bool gemm_n_given = false;
    bool gemm_iters_given = false;
    bool gemm_tf32_given = false;
    bool fft_n_given = false;
    bool fft_batch_given = false;
    bool fft_iters_given = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
            out_dir_given = true;
        } else if (arg == "--tag" && i + 1 < argc) {
            tag = argv[++i];
        } else if (arg == "--energy-duration-ms" && i + 1 < argc) {
            energy_duration_given = parse_int_arg(argv[++i], &energy_duration_ms);
            if (!energy_duration_given || energy_duration_ms <= 0) {
                std::fprintf(stderr, "Invalid value for --energy-duration-ms\n");
                return 1;
            }
        } else if (arg == "--energy-meta-out" && i + 1 < argc) {
            energy_meta_out = argv[++i];
            energy_meta_out_given = true;
        } else if (arg == "--bw-bytes" && i + 1 < argc) {
            bw_bytes_given = parse_size_arg(argv[++i], &bw_bytes);
            if (!bw_bytes_given) {
                std::fprintf(stderr, "Invalid value for --bw-bytes\n");
                return 1;
            }
        } else if (arg == "--bw-iters" && i + 1 < argc) {
            bw_iters_given = parse_int_arg(argv[++i], &bw_iters);
            if (!bw_iters_given) {
                std::fprintf(stderr, "Invalid value for --bw-iters\n");
                return 1;
            }
        } else if (arg == "--bw-block" && i + 1 < argc) {
            bw_block_given = parse_int_arg(argv[++i], &bw_block);
            if (!bw_block_given) {
                std::fprintf(stderr, "Invalid value for --bw-block\n");
                return 1;
            }
        } else if (arg == "--compute-dtype" && i + 1 < argc) {
            compute_dtype = argv[++i];
            compute_dtype_given = true;
        } else if (arg == "--compute-block" && i + 1 < argc) {
            compute_block_given = parse_int_arg(argv[++i], &compute_block);
            if (!compute_block_given) {
                std::fprintf(stderr, "Invalid value for --compute-block\n");
                return 1;
            }
        } else if (arg == "--compute-grid" && i + 1 < argc) {
            compute_grid_given = parse_int_arg(argv[++i], &compute_grid);
            if (!compute_grid_given) {
                std::fprintf(stderr, "Invalid value for --compute-grid\n");
                return 1;
            }
        } else if (arg == "--compute-iters" && i + 1 < argc) {
            compute_iters_given = parse_int_arg(argv[++i], &compute_iters);
            if (!compute_iters_given) {
                std::fprintf(stderr, "Invalid value for --compute-iters\n");
                return 1;
            }
        } else if (arg == "--gemm-n" && i + 1 < argc) {
            gemm_n_given = parse_int_arg(argv[++i], &gemm_n);
            if (!gemm_n_given) {
                std::fprintf(stderr, "Invalid value for --gemm-n\n");
                return 1;
            }
        } else if (arg == "--gemm-iters" && i + 1 < argc) {
            gemm_iters_given = parse_int_arg(argv[++i], &gemm_iters);
            if (!gemm_iters_given) {
                std::fprintf(stderr, "Invalid value for --gemm-iters\n");
                return 1;
            }
        } else if (arg == "--gemm-tf32" && i + 1 < argc) {
            gemm_tf32_given = parse_int_arg(argv[++i], &gemm_tf32);
            if (!gemm_tf32_given) {
                std::fprintf(stderr, "Invalid value for --gemm-tf32\n");
                return 1;
            }
        } else if (arg == "--fft-n" && i + 1 < argc) {
            fft_n_given = parse_int_arg(argv[++i], &fft_n);
            if (!fft_n_given) {
                std::fprintf(stderr, "Invalid value for --fft-n\n");
                return 1;
            }
        } else if (arg == "--fft-batch" && i + 1 < argc) {
            fft_batch_given = parse_int_arg(argv[++i], &fft_batch);
            if (!fft_batch_given) {
                std::fprintf(stderr, "Invalid value for --fft-batch\n");
                return 1;
            }
        } else if (arg == "--fft-iters" && i + 1 < argc) {
            fft_iters_given = parse_int_arg(argv[++i], &fft_iters);
            if (!fft_iters_given) {
                std::fprintf(stderr, "Invalid value for --fft-iters\n");
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!valid_mode(mode)) {
        std::fprintf(stderr, "Invalid mode '%s'\n", mode.c_str());
        usage(argv[0]);
        return 1;
    }

    const bool energy_mode = energy_duration_given || energy_meta_out_given;

    const bool bw_case = bw_bytes_given || bw_iters_given || bw_block_given;
    if (bw_case) {
        if (mode != "bw") {
            std::fprintf(stderr, "BW case flags require --mode bw\n");
            return 1;
        }
        if (!(bw_bytes_given && bw_iters_given && bw_block_given)) {
            std::fprintf(stderr, "BW case requires --bw-bytes, --bw-iters and --bw-block together\n");
            return 1;
        }
    }

    const bool compute_case = compute_dtype_given || compute_block_given || compute_grid_given || compute_iters_given;
    if (compute_case) {
        if (mode != "compute") {
            std::fprintf(stderr, "Compute case flags require --mode compute\n");
            return 1;
        }
        if (!(compute_dtype_given && compute_block_given && compute_grid_given && compute_iters_given)) {
            std::fprintf(stderr,
                         "Compute case requires --compute-dtype, --compute-block, --compute-grid and --compute-iters together\n");
            return 1;
        }
        if (!valid_compute_dtype(compute_dtype)) {
            std::fprintf(stderr, "Invalid compute dtype '%s'\n", compute_dtype.c_str());
            return 1;
        }
    }

    const bool gemm_case = gemm_n_given || gemm_iters_given || gemm_tf32_given;
    if (gemm_case) {
        if (mode != "gemm") {
            std::fprintf(stderr, "GEMM case flags require --mode gemm\n");
            return 1;
        }
        if (!(gemm_n_given && gemm_iters_given && gemm_tf32_given)) {
            std::fprintf(stderr, "GEMM case requires --gemm-n, --gemm-iters and --gemm-tf32 together\n");
            return 1;
        }
        if (gemm_tf32 != 0 && gemm_tf32 != 1) {
            std::fprintf(stderr, "Invalid --gemm-tf32 value %d (expected 0 or 1)\n", gemm_tf32);
            return 1;
        }
    }

    const bool fft_case = fft_n_given || fft_batch_given || fft_iters_given;
    if (fft_case) {
        if (mode != "fft") {
            std::fprintf(stderr, "FFT case flags require --mode fft\n");
            return 1;
        }
        if (!(fft_n_given && fft_batch_given && fft_iters_given)) {
            std::fprintf(stderr, "FFT case requires --fft-n, --fft-batch and --fft-iters together\n");
            return 1;
        }
        if (fft_n <= 0 || fft_batch <= 0 || fft_iters <= 0) {
            std::fprintf(stderr, "FFT case values must be > 0\n");
            return 1;
        }
    }

    if (energy_mode) {
        if (!(energy_duration_given && energy_meta_out_given)) {
            std::fprintf(stderr, "Energy mode requires both --energy-duration-ms and --energy-meta-out\n");
            return 1;
        }
        if (mode == "all") {
            std::fprintf(stderr, "Energy mode is only supported for case-specific bw/compute/gemm executions\n");
            return 1;
        }
        if ((mode == "bw" && !bw_case) || (mode == "compute" && !compute_case) ||
            (mode == "gemm" && !gemm_case) || (mode == "fft" && !fft_case)) {
            std::fprintf(stderr, "Energy mode requires a fully specified benchmark case\n");
            return 1;
        }
    }

    if (!out_dir_given) {
        const std::string env_folder = detect_env_folder();
        out_dir = join_path(join_path("results", env_folder), "baseline");
        std::printf("Auto output dir: %s\n", out_dir.c_str());
    }

    std::string base_out = out_dir;
    if (!tag.empty()) base_out = join_path(base_out, tag);

    std::string mkdir_err;
    if (!ensure_dir_recursive(base_out, &mkdir_err)) {
        std::fprintf(stderr, "Cannot create output directory '%s': %s\n", base_out.c_str(), mkdir_err.c_str());
        return 1;
    }

    if (energy_mode) {
        const std::string meta_parent = parent_dir(energy_meta_out);
        if (!ensure_dir_recursive(meta_parent, &mkdir_err)) {
            std::fprintf(stderr, "Cannot create metadata directory '%s': %s\n", meta_parent.c_str(), mkdir_err.c_str());
            return 1;
        }
    }

    print_device_info();

    const std::string bw_path = join_path(base_out, "bw.csv");
    const std::string compute_path = join_path(base_out, "compute.csv");
    const std::string gemm_path = join_path(base_out, "gemm.csv");
    const std::string fft_path = join_path(base_out, "fft.csv");

    if (energy_mode) {
        std::string case_key;
        std::string params;
        std::string perf_unit;
        EnergyRunResult result{};

        const auto run_start = std::chrono::system_clock::now();
        if (mode == "bw") {
            case_key = "bw";
            params = "--bw-bytes " + std::to_string(bw_bytes) +
                     " --bw-iters " + std::to_string(bw_iters) +
                     " --bw-block " + std::to_string(bw_block);
            perf_unit = "GB/s";
            result = run_bw_case_for_energy(bw_bytes, bw_iters, bw_block, static_cast<double>(energy_duration_ms));
            write_bw_result_csv(bw_path, bw_bytes, bw_iters, bw_block, result.avg_perf);
            std::printf("BW long-run avg -> %.3f GB/s\n", result.avg_perf);
            std::printf("Wrote %s\n", bw_path.c_str());
        } else if (mode == "compute") {
            case_key = "compute";
            params = "--compute-dtype " + compute_dtype +
                     " --compute-block " + std::to_string(compute_block) +
                     " --compute-grid " + std::to_string(compute_grid) +
                     " --compute-iters " + std::to_string(compute_iters);
            perf_unit = "GFLOP/s";
            result = run_compute_case_for_energy(compute_dtype.c_str(), compute_block, compute_grid, compute_iters,
                                                 static_cast<double>(energy_duration_ms));
            write_compute_result_csv(compute_path, compute_dtype, compute_block, compute_grid, compute_iters, result.avg_perf);
            std::printf("%s long-run avg -> %.2f GFLOP/s\n", compute_dtype.c_str(), result.avg_perf);
            std::printf("Wrote %s\n", compute_path.c_str());
        } else if (mode == "gemm") {
            case_key = "gemm";
            params = "--gemm-n " + std::to_string(gemm_n) +
                     " --gemm-iters " + std::to_string(gemm_iters) +
                     " --gemm-tf32 " + std::to_string(gemm_tf32);
            perf_unit = "GFLOP/s";
            result = run_gemm_case_for_energy(gemm_n, gemm_iters, gemm_tf32 == 1, static_cast<double>(energy_duration_ms));
            write_gemm_result_csv(gemm_path, gemm_n, gemm_iters, gemm_tf32 == 1, result.avg_perf);
            std::printf("GEMM long-run avg -> %.2f GFLOP/s\n", result.avg_perf);
            std::printf("Wrote %s\n", gemm_path.c_str());
        } else if (mode == "fft") {
            case_key = "fft";
            params = "--fft-n " + std::to_string(fft_n) +
                     " --fft-batch " + std::to_string(fft_batch) +
                     " --fft-iters " + std::to_string(fft_iters);
            perf_unit = "MSamples/s";
            result = run_fft_case_for_energy(fft_n, fft_batch, fft_iters, static_cast<double>(energy_duration_ms));
            const double exec_calls = static_cast<double>(fft_iters) * static_cast<double>(result.case_repeats);
            const double time_ms = result.measured_work_ms / std::max(exec_calls, 1.0);
            const double transforms_per_s = (result.avg_perf * 1e6) / static_cast<double>(fft_n);
            write_fft_result_csv(fft_path, fft_n, fft_batch, fft_iters, time_ms, transforms_per_s, result.avg_perf);
            std::printf("FFT long-run avg -> %.2f MSamples/s\n", result.avg_perf);
            std::printf("Wrote %s\n", fft_path.c_str());
        }
        const auto run_end = std::chrono::system_clock::now();

        const double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(run_end - run_start).count();
        write_energy_meta_csv(energy_meta_out,
                              case_key,
                              params,
                              perf_unit,
                              result.avg_perf,
                              static_cast<double>(energy_duration_ms),
                              result.measured_work_ms,
                              wall_ms,
                              result.case_repeats,
                              format_utc_timestamp(run_start),
                              format_utc_timestamp(run_end));
        std::printf("Wrote %s\n", energy_meta_out.c_str());
        return 0;
    }

    if (mode == "bw" || mode == "all") {
        if (bw_case) {
            run_bw_case_to_csv(bw_path.c_str(), bw_bytes, bw_iters, bw_block);
        } else {
            run_bw_sweep_to_csv(bw_path.c_str());
        }
        std::printf("Wrote %s\n", bw_path.c_str());
    }
    if (mode == "compute" || mode == "all") {
        if (compute_case) {
            run_compute_case_to_csv(compute_path.c_str(), compute_dtype.c_str(), compute_block, compute_grid, compute_iters);
        } else {
            run_compute_sweep_to_csv(compute_path.c_str());
        }
        std::printf("Wrote %s\n", compute_path.c_str());
    }
    if (mode == "gemm" || mode == "all") {
        if (gemm_case) {
            run_gemm_case_to_csv(gemm_path.c_str(), gemm_n, gemm_iters, gemm_tf32 == 1);
        } else {
            run_gemm_sweep_to_csv(gemm_path.c_str());
        }
        std::printf("Wrote %s\n", gemm_path.c_str());
    }
    if (mode == "fft" || mode == "all") {
        if (fft_case) {
            run_fft_case_to_csv(fft_path.c_str(), fft_n, fft_batch, fft_iters);
        } else {
            run_fft_sweep_to_csv(fft_path.c_str());
        }
        std::printf("Wrote %s\n", fft_path.c_str());
    }

    return 0;
}
