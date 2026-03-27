from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from perf_targets import TEST_ORDER

OPTIONAL_PERFORMANCE_TESTS = {"FFT C2C max"}

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
EXPECTED_UNITS = {
    "performance": {"GB/s", "GFLOP/s", "MSamples/s"},
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

    cols = set(df.columns)
    if not REQUIRED_COLUMNS.issubset(cols):
        fail(f"{path}: missing required columns {sorted(REQUIRED_COLUMNS.difference(cols))}, got {list(df.columns)}")

    metrics = set(df["metric"].astype(str))
    if metrics != EXPECTED_METRICS:
        fail(f"{path}: unexpected metric set {sorted(metrics)}")

    perf_df = df[df["metric"].astype(str) == "performance"].copy()
    eff_df = df[df["metric"].astype(str) == "efficiency"].copy()

    perf_tests = set(perf_df["test"].astype(str))
    eff_tests = set(eff_df["test"].astype(str))
    allowed_perf_tests = EXPECTED_TESTS.union(OPTIONAL_PERFORMANCE_TESTS)
    if not EXPECTED_TESTS.issubset(perf_tests) or not perf_tests.issubset(allowed_perf_tests):
        fail(f"{path}: unexpected performance test set {sorted(perf_tests)}")
    if eff_tests != EXPECTED_TESTS:
        fail(f"{path}: unexpected efficiency test set {sorted(eff_tests)}")

    perf_units = set(perf_df["unit"].astype(str))
    eff_units = set(eff_df["unit"].astype(str))
    if not perf_units.issubset(EXPECTED_UNITS["performance"]):
        fail(f"{path}: unexpected performance units {sorted(perf_units)}")
    if eff_units != EXPECTED_UNITS["efficiency"]:
        fail(f"{path}: unexpected efficiency units {sorted(eff_units)}")

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
    if not ENV_COMPARE_COLUMNS.issubset(cols):
        fail(f"{path}: missing required columns {sorted(ENV_COMPARE_COLUMNS)}, got {list(df.columns)}")

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
        "energy_duration_ms",
        "stable_window_trim",
        "power_telemetry_source",
        "power_scope",
        "node_power_not_measured",
        "activity_definition",
    }
    fields = set(df["field"].astype(str))
    if not required_fields.issubset(fields):
        fail(f"{path}: missing metadata fields {sorted(required_fields.difference(fields))}")


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)

    check_files(compare_dir)
    check_summary(compare_dir / "summary_compare.csv")
    check_env_compare(compare_dir / "environment_compare.csv")

    print(f"Comparison artifacts OK: {compare_dir}")


if __name__ == "__main__":
    main()
