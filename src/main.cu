#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>

void print_device_info();

void run_bw_sweep_to_csv(const char* path);
void run_bw_case_to_csv(const char* path, size_t bytes, int iters, int block);
void run_compute_sweep_to_csv(const char* path);
void run_compute_case_to_csv(const char* path, const char* dtype, int block, int grid, int iters);
void run_gemm_sweep_to_csv(const char* path);
void run_gemm_case_to_csv(const char* path, int N, int iters, bool tf32);

static bool valid_mode(const std::string& mode) {
    return mode == "all" || mode == "bw" || mode == "compute" || mode == "gemm";
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

static bool ensure_dir_recursive(const std::string& dir, std::string* err) {
    if (dir.empty()) {
        if (err) *err = "empty path";
        return false;
    }

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

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [--mode all|bw|compute|gemm] [--out-dir PATH] [--tag NAME] [advanced case flags]\n"
        "Default out-dir: auto -> results/<a100|rtx5000|other_gpu>/baseline\n"
        "Advanced case flags:\n"
        "  BW:      --bw-bytes N --bw-iters N --bw-block N\n"
        "  Compute: --compute-dtype fp32|fp64 --compute-block N --compute-grid N --compute-iters N\n"
        "  GEMM:    --gemm-n N --gemm-iters N --gemm-tf32 0|1\n"
        "Examples:\n"
        "  %s --mode all\n"
        "  %s --mode gemm --out-dir results/a100/baseline\n"
        "  %s --mode compute --compute-dtype fp32 --compute-block 1024 --compute-grid 864 --compute-iters 4096\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string mode = "all";
    std::string out_dir;
    std::string tag;
    bool out_dir_given = false;
    size_t bw_bytes = 0;
    int bw_iters = 0;
    int bw_block = 0;
    bool bw_bytes_given = false;
    bool bw_iters_given = false;
    bool bw_block_given = false;
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
    bool gemm_n_given = false;
    bool gemm_iters_given = false;
    bool gemm_tf32_given = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
            out_dir_given = true;
        } else if (arg == "--tag" && i + 1 < argc) {
            tag = argv[++i];
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

    print_device_info();

    const std::string bw_path = join_path(base_out, "bw.csv");
    const std::string compute_path = join_path(base_out, "compute.csv");
    const std::string gemm_path = join_path(base_out, "gemm.csv");

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

    return 0;
}
