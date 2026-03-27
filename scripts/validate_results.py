from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

from perf_targets import TEST_ORDER, load_perf_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark CSV outputs")
    parser.add_argument("--results-dir", default="results/a100/baseline", help="Directory containing bw.csv, compute.csv and gemm.csv")
    parser.add_argument("--require-energy", action="store_true",
                        help="Require an energy summary at <results-dir>/energy/efficiency_active_summary.csv")
    parser.add_argument("--energy-summary", default="",
                        help="Optional explicit path to efficiency_active_summary.csv")
    return parser.parse_args()


def fail(msg: str) -> None:
    raise ValueError(msg)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"Missing file: {path}")
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"No data rows in {path}")
    return rows


def to_float(path: Path, row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:
        fail(f"{path}: cannot parse '{key}' value '{row.get(key)}' ({exc})")


def to_int(path: Path, row: dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception as exc:
        fail(f"{path}: cannot parse '{key}' value '{row.get(key)}' ({exc})")


def to_bool(path: Path, row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    fail(f"{path}: cannot parse '{key}' boolean value '{row.get(key)}'")


def expect_close(path: Path, label: str, actual: float, expected: float, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        fail(f"{path}: unexpected {label}: got {actual}, expected {expected}")


def validate_bw(path: Path) -> None:
    rows = read_rows(path)
    required = {"bytes", "iters", "block", "GBs"}
    if set(rows[0].keys()) != required:
        fail(f"{path}: expected columns {sorted(required)}, got {list(rows[0].keys())}")

    prev_bytes = -1
    for r in rows:
        b = to_int(path, r, "bytes")
        g = to_float(path, r, "GBs")
        if b <= 0:
            fail(f"{path}: non-positive bytes {b}")
        if g <= 0:
            fail(f"{path}: non-positive GB/s {g}")
        if b <= prev_bytes:
            fail(f"{path}: bytes are not strictly increasing")
        prev_bytes = b


def validate_compute(path: Path) -> None:
    rows = read_rows(path)
    required = {"dtype", "block", "grid", "iters", "GFLOPs"}
    if set(rows[0].keys()) != required:
        fail(f"{path}: expected columns {sorted(required)}, got {list(rows[0].keys())}")

    dtypes = set()
    for r in rows:
        dtype = r["dtype"].strip()
        dtypes.add(dtype)
        if dtype not in {"fp32", "fp64"}:
            fail(f"{path}: unexpected dtype '{dtype}'")
        if to_int(path, r, "block") <= 0:
            fail(f"{path}: block must be > 0")
        if to_int(path, r, "grid") <= 0:
            fail(f"{path}: grid must be > 0")
        if to_int(path, r, "iters") <= 0:
            fail(f"{path}: iters must be > 0")
        if to_float(path, r, "GFLOPs") <= 0:
            fail(f"{path}: GFLOPs must be > 0")

    if dtypes != {"fp32", "fp64"}:
        fail(f"{path}: expected both fp32 and fp64, got {sorted(dtypes)}")


def validate_gemm(path: Path) -> None:
    rows = read_rows(path)
    required = {"N", "iters", "tf32", "GFLOPs"}
    if set(rows[0].keys()) != required:
        fail(f"{path}: expected columns {sorted(required)}, got {list(rows[0].keys())}")

    tf32_values = set()
    for r in rows:
        n = to_int(path, r, "N")
        t = to_int(path, r, "tf32")
        g = to_float(path, r, "GFLOPs")
        if n <= 0:
            fail(f"{path}: N must be > 0")
        if t not in {0, 1}:
            fail(f"{path}: tf32 must be 0/1, got {t}")
        if g <= 0:
            fail(f"{path}: GFLOPs must be > 0")
        tf32_values.add(t)

    if tf32_values != {0, 1}:
        fail(f"{path}: expected both tf32=0 and tf32=1")


def validate_fft(path: Path) -> None:
    rows = read_rows(path)
    required = {"n", "batch", "iters", "time_ms", "transforms_per_s", "MSamples_per_s"}
    if set(rows[0].keys()) != required:
        fail(f"{path}: expected columns {sorted(required)}, got {list(rows[0].keys())}")

    prev_n = -1
    for r in rows:
        n = to_int(path, r, "n")
        batch = to_int(path, r, "batch")
        iters = to_int(path, r, "iters")
        time_ms = to_float(path, r, "time_ms")
        transforms = to_float(path, r, "transforms_per_s")
        msamples = to_float(path, r, "MSamples_per_s")
        if n <= 0:
            fail(f"{path}: n must be > 0")
        if batch <= 0:
            fail(f"{path}: batch must be > 0")
        if iters <= 0:
            fail(f"{path}: iters must be > 0")
        if time_ms <= 0:
            fail(f"{path}: time_ms must be > 0")
        if transforms <= 0:
            fail(f"{path}: transforms_per_s must be > 0")
        if msamples <= 0:
            fail(f"{path}: MSamples_per_s must be > 0")
        if n <= prev_n:
            fail(f"{path}: n values must be strictly increasing")
        prev_n = n


def validate_energy(path: Path, perf_dir: Path) -> None:
    rows = read_rows(path)

    modern = {
        "Test",
        "Perf unit",
        "Perf",
        "Active power (W)",
        "Efficiency (unit/W)",
        "Baseline perf",
        "Run duration (ms)",
        "Window duration (ms)",
        "Window samples",
        "Trim ratio",
        "Fallback window",
        "Mean SM in window",
        "Peak SM in window",
        "Run samples",
        "Measured work (ms)",
        "Target duration (ms)",
        "Case repeats",
        "Perf delta vs baseline (%)",
        "SM active threshold (MHz)",
        "Window samples above SM threshold (%)",
        "Fallback run window",
        "Meta case key",
        "Power log",
        "Meta log",
    }
    legacy = {
        "Test",
        "Perf unit",
        "Perf (baseline)",
        "Active power (W)",
        "Efficiency (unit/W)",
    }

    cols = set(rows[0].keys())
    if modern.issubset(cols):
        perf_key = "Perf"
        need_samples = True
        targets = load_perf_targets(perf_dir)
        expected_tests_order = [target.test for target in targets]
        target_by_test = {target.test: target for target in targets}
        expected_tests = set(expected_tests_order)
        expected_units = {target.test: target.perf_unit for target in targets}
    elif legacy.issubset(cols):
        perf_key = "Perf (baseline)"
        need_samples = False
        targets = []
        target_by_test = {}
        expected_tests_order = TEST_ORDER
        expected_tests = {
            "BW plateau",
            "Compute FP32 peak",
            "Compute FP64 peak",
            "GEMM TF32=0 max",
            "GEMM TF32=1 max",
        }
        expected_units = {
            "BW plateau": "GB/s",
            "Compute FP32 peak": "GFLOP/s",
            "Compute FP64 peak": "GFLOP/s",
            "GEMM TF32=0 max": "GFLOP/s",
            "GEMM TF32=1 max": "GFLOP/s",
        }
    else:
        fail(f"{path}: missing required modern columns {sorted(modern)} and legacy columns {sorted(legacy)}")

    tests = [r["Test"].strip() for r in rows]
    if set(tests) != expected_tests or len(tests) != len(expected_tests):
        fail(f"{path}: unexpected test rows {tests}; expected exactly {sorted(expected_tests)}")
    if need_samples and tests != expected_tests_order:
        fail(f"{path}: unexpected test order {tests}; expected {expected_tests_order}")

    for r in rows:
        test = r["Test"].strip()
        if r["Perf unit"].strip() != expected_units[test]:
            fail(f"{path}: unexpected unit for '{test}': {r['Perf unit']}")
        perf_value = to_float(path, r, perf_key)
        active_power = to_float(path, r, "Active power (W)")
        efficiency = to_float(path, r, "Efficiency (unit/W)")
        if perf_value <= 0:
            fail(f"{path}: {perf_key} must be > 0")
        if active_power <= 0:
            fail(f"{path}: Active power must be > 0")
        if efficiency <= 0:
            fail(f"{path}: Efficiency must be > 0")
        expect_close(path, f"efficiency for '{test}'", efficiency, perf_value / active_power)
        if need_samples:
            target = target_by_test[test]
            baseline_perf = to_float(path, r, "Baseline perf")
            run_duration = to_float(path, r, "Run duration (ms)")
            window_duration = to_float(path, r, "Window duration (ms)")
            window_samples = to_int(path, r, "Window samples")
            run_samples = to_int(path, r, "Run samples")
            trim_ratio = to_float(path, r, "Trim ratio")
            mean_sm = to_float(path, r, "Mean SM in window")
            peak_sm = to_float(path, r, "Peak SM in window")
            measured_work = to_float(path, r, "Measured work (ms)")
            target_duration = to_float(path, r, "Target duration (ms)")
            case_repeats = to_int(path, r, "Case repeats")
            perf_delta_pct = to_float(path, r, "Perf delta vs baseline (%)")
            sm_threshold = to_float(path, r, "SM active threshold (MHz)")
            active_pct = to_float(path, r, "Window samples above SM threshold (%)")
            fallback_window = to_bool(path, r, "Fallback window")
            fallback_run_window = to_bool(path, r, "Fallback run window")
            meta_case_key = r["Meta case key"].strip()
            power_log = r["Power log"].strip()
            meta_log = r["Meta log"].strip()

            expect_close(path, f"baseline perf for '{test}'", baseline_perf, float(target.perf))
            if meta_case_key != target.case_key:
                fail(f"{path}: unexpected Meta case key for '{test}': {meta_case_key}")
            if power_log != target.power_log_name:
                fail(f"{path}: unexpected Power log for '{test}': {power_log}")
            if meta_log != target.meta_log_name:
                fail(f"{path}: unexpected Meta log for '{test}': {meta_log}")

            if run_duration <= 0:
                fail(f"{path}: Run duration (ms) must be > 0")
            if window_duration < 0:
                fail(f"{path}: Window duration (ms) must be >= 0")
            if window_samples <= 0:
                fail(f"{path}: Window samples must be > 0")
            if run_samples < window_samples:
                fail(f"{path}: Run samples must be >= Window samples for '{test}'")
            if measured_work <= 0:
                fail(f"{path}: Measured work (ms) must be > 0")
            if target_duration <= 0:
                fail(f"{path}: Target duration (ms) must be > 0")
            if measured_work + 1e-6 < target_duration:
                fail(f"{path}: Measured work (ms) must be >= Target duration (ms) for '{test}'")
            if run_duration + 5.0 < measured_work:
                fail(f"{path}: Run duration (ms) must not be materially smaller than Measured work (ms) for '{test}'")
            if case_repeats <= 0:
                fail(f"{path}: Case repeats must be > 0")
            if trim_ratio < 0 or trim_ratio >= 0.5:
                fail(f"{path}: Trim ratio must be in [0, 0.5)")
            if mean_sm <= 0:
                fail(f"{path}: Mean SM in window must be > 0")
            if peak_sm <= 0:
                fail(f"{path}: Peak SM in window must be > 0")
            if mean_sm > peak_sm + 1e-6:
                fail(f"{path}: Mean SM in window cannot exceed Peak SM in window for '{test}'")
            if sm_threshold <= 0:
                fail(f"{path}: SM active threshold (MHz) must be > 0")
            if active_pct < 0 or active_pct > 100:
                fail(f"{path}: Window samples above SM threshold (%) must be in [0, 100]")
            if fallback_window and window_samples != run_samples:
                fail(f"{path}: Fallback window rows must use the full run window for '{test}'")
            if fallback_window and trim_ratio != 0:
                fail(f"{path}: Trim ratio must be 0 when Fallback window is true for '{test}'")
            if fallback_run_window:
                fail(f"{path}: Fallback run window should be false for the current methodology ('{test}')")
            expect_close(
                path,
                f"Perf delta vs baseline (%) for '{test}'",
                perf_delta_pct,
                ((perf_value - baseline_perf) / baseline_perf) * 100.0,
                rel_tol=1e-6,
                abs_tol=1e-4,
            )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    validate_bw(results_dir / "bw.csv")
    validate_compute(results_dir / "compute.csv")
    validate_gemm(results_dir / "gemm.csv")
    fft_path = results_dir / "fft.csv"
    if fft_path.exists():
        validate_fft(fft_path)

    energy_summary = None
    if args.energy_summary:
        energy_summary = Path(args.energy_summary)
    elif args.require_energy:
        if results_dir.name == "baseline":
            energy_summary = results_dir.parent / "energy" / "efficiency_active_summary.csv"
        else:
            energy_summary = results_dir / "energy" / "efficiency_active_summary.csv"

    if energy_summary is not None:
        validate_energy(energy_summary, results_dir)

    print(f"Validation OK for {results_dir}")
    if energy_summary is not None:
        print(f"Energy summary OK: {energy_summary}")


if __name__ == "__main__":
    main()
