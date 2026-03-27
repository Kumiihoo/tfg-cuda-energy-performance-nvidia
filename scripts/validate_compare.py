from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from perf_targets import TEST_ORDER


REQUIRED_FILES = [
    "summary_compare.csv",
    "environment_compare.csv",
    "methodology_notes.txt",
    "perf_absolute_compare.png",
    "efficiency_compare.png",
    "speedup_a100_vs_rtx5000.png",
]

REQUIRED_COLUMNS = {
    "test",
    "metric",
    "unit",
    "a100",
    "rtx5000",
    "ratio_a100_vs_rtx5000",
    "delta_percent",
}

EXPECTED_TESTS = set(TEST_ORDER)

EXPECTED_METRICS = {"performance", "efficiency"}
EXPECTED_ROW_COUNT = len(EXPECTED_TESTS) * len(EXPECTED_METRICS)
EXPECTED_UNITS = {
    "performance": {"GB/s", "GFLOP/s"},
    "efficiency": {"GB/s/W", "GFLOP/s/W"},
}
ENV_COMPARE_COLUMNS = {
    "category",
    "field",
    "a100",
    "rtx5000",
    "match",
    "expected_match",
    "status",
}
ENV_COMPARE_STATUSES = {"ok", "info", "warning"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate A100 vs RTX5000 comparison artifacts")
    parser.add_argument("--compare-dir", default="results/compare", help="Directory with summary_compare.csv and comparison plots")
    return parser.parse_args()


def fail(msg: str) -> None:
    raise ValueError(msg)


def check_files(compare_dir: Path) -> None:
    for name in REQUIRED_FILES:
        path = compare_dir / name
        if not path.exists():
            fail(f"Missing file: {path}")
        if path.stat().st_size <= 0:
            fail(f"Empty file: {path}")


def check_summary(path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        fail(f"No rows in {path}")
    if len(df) != EXPECTED_ROW_COUNT:
        fail(f"{path}: expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    cols = set(df.columns)
    if cols != REQUIRED_COLUMNS:
        fail(f"{path}: expected columns {sorted(REQUIRED_COLUMNS)}, got {list(df.columns)}")

    tests = set(df["test"].astype(str))
    if tests != EXPECTED_TESTS:
        fail(f"{path}: unexpected test set {sorted(tests)}")

    metrics = set(df["metric"].astype(str))
    if metrics != EXPECTED_METRICS:
        fail(f"{path}: unexpected metric set {sorted(metrics)}")

    for metric, expected_units in EXPECTED_UNITS.items():
        units = set(df.loc[df["metric"].astype(str) == metric, "unit"].astype(str))
        if units != expected_units:
            fail(f"{path}: unexpected units for metric '{metric}': {sorted(units)}")

    for col in ["a100", "rtx5000", "ratio_a100_vs_rtx5000"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            fail(f"{path}: non-numeric values in {col}")
        if (values <= 0).any():
            fail(f"{path}: non-positive values in {col}")


def check_env_compare(path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        fail(f"No rows in {path}")

    cols = set(df.columns)
    if cols != ENV_COMPARE_COLUMNS:
        fail(f"{path}: expected columns {sorted(ENV_COMPARE_COLUMNS)}, got {list(df.columns)}")

    statuses = set(df["status"].astype(str))
    if not statuses.issubset(ENV_COMPARE_STATUSES):
        fail(f"{path}: unexpected status values {sorted(statuses)}")

    required_fields = {
        "environment",
        "host",
        "gpu_name",
        "git_commit",
        "driver_version",
        "cuda_driver_version",
        "nvcc_version",
        "python_version",
        "sample_ms",
        "bw_repeats",
        "compute_repeats",
        "gemm_repeats",
        "power_telemetry_source",
        "power_scope",
        "node_power_not_measured",
    }
    fields = set(df["field"].astype(str))
    if fields != required_fields:
        fail(f"{path}: unexpected metadata field set {sorted(fields)}")


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)

    check_files(compare_dir)
    check_summary(compare_dir / "summary_compare.csv")
    check_env_compare(compare_dir / "environment_compare.csv")

    print(f"Comparison artifacts OK: {compare_dir}")


if __name__ == "__main__":
    main()
