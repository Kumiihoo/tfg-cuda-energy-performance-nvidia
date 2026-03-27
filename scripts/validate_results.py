from __future__ import annotations
import argparse
import csv
from pathlib import Path

from perf_targets import TEST_ORDER


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


def validate_energy(path: Path) -> None:
    rows = read_rows(path)

    modern = {
        "Test",
        "Perf unit",
        "Perf",
        "Active power (W)",
        "Efficiency (unit/W)",
        "Active threshold (MHz)",
        "Active samples (%)",
        "Samples",
        "Peak SM observed (MHz)",
        "Fallback all samples",
    }
    legacy = {
        "Test",
        "Perf unit",
        "Perf (baseline)",
        "Active power (W)",
        "Efficiency (unit/W)",
    }

    cols = set(rows[0].keys())
    if cols == modern:
        perf_key = "Perf"
        need_samples = True
        expected_tests = set(TEST_ORDER)
        expected_units = {
            "BW peak": "GB/s",
            "BW sustained": "GB/s",
            "Compute FP32 peak": "GFLOP/s",
            "Compute FP64 peak": "GFLOP/s",
            "GEMM TF32=0 max": "GFLOP/s",
            "GEMM TF32=1 max": "GFLOP/s",
        }
    elif cols == legacy:
        perf_key = "Perf (baseline)"
        need_samples = False
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
        fail(f"{path}: expected columns {sorted(modern)} or {sorted(legacy)}, got {list(rows[0].keys())}")

    tests = [r["Test"].strip() for r in rows]
    if set(tests) != expected_tests or len(tests) != len(expected_tests):
        fail(f"{path}: unexpected test rows {tests}; expected exactly {sorted(expected_tests)}")

    for r in rows:
        test = r["Test"].strip()
        if r["Perf unit"].strip() != expected_units[test]:
            fail(f"{path}: unexpected unit for '{test}': {r['Perf unit']}")
        if to_float(path, r, perf_key) <= 0:
            fail(f"{path}: {perf_key} must be > 0")
        if to_float(path, r, "Active power (W)") <= 0:
            fail(f"{path}: Active power must be > 0")
        if to_float(path, r, "Efficiency (unit/W)") <= 0:
            fail(f"{path}: Efficiency must be > 0")
        if need_samples and to_int(path, r, "Samples") <= 0:
            fail(f"{path}: Samples must be > 0")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    validate_bw(results_dir / "bw.csv")
    validate_compute(results_dir / "compute.csv")
    validate_gemm(results_dir / "gemm.csv")

    energy_summary = None
    if args.energy_summary:
        energy_summary = Path(args.energy_summary)
    elif args.require_energy:
        if results_dir.name == "baseline":
            energy_summary = results_dir.parent / "energy" / "efficiency_active_summary.csv"
        else:
            energy_summary = results_dir / "energy" / "efficiency_active_summary.csv"

    if energy_summary is not None:
        validate_energy(energy_summary)

    print(f"Validation OK for {results_dir}")
    if energy_summary is not None:
        print(f"Energy summary OK: {energy_summary}")


if __name__ == "__main__":
    main()
