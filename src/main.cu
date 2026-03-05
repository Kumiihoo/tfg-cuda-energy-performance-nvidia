#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>

void print_device_info();

void run_bw_sweep_to_csv(const char* path);
void run_compute_sweep_to_csv(const char* path);
void run_gemm_sweep_to_csv(const char* path);

static bool valid_mode(const std::string& mode) {
    return mode == "all" || mode == "bw" || mode == "compute" || mode == "gemm";
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
        "Usage: %s [--mode all|bw|compute|gemm] [--out-dir PATH] [--tag NAME]\n"
        "Default out-dir: auto -> results/<a100|rtx5000|other_gpu>/baseline\n"
        "Examples:\n"
        "  %s --mode all\n"
        "  %s --mode gemm --out-dir results/a100/baseline\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string mode = "all";
    std::string out_dir;
    std::string tag;
    bool out_dir_given = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
            out_dir_given = true;
        } else if (arg == "--tag" && i + 1 < argc) {
            tag = argv[++i];
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
        run_bw_sweep_to_csv(bw_path.c_str());
        std::printf("Wrote %s\n", bw_path.c_str());
    }
    if (mode == "compute" || mode == "all") {
        run_compute_sweep_to_csv(compute_path.c_str());
        std::printf("Wrote %s\n", compute_path.c_str());
    }
    if (mode == "gemm" || mode == "all") {
        run_gemm_sweep_to_csv(gemm_path.c_str());
        std::printf("Wrote %s\n", gemm_path.c_str());
    }

    return 0;
}


