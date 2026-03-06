from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "summary_compare.csv",
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

EXPECTED_TESTS = {
    "BW plateau",
    "Compute FP32 peak",
    "Compute FP64 peak",
    "GEMM TF32=0 max",
    "GEMM TF32=1 max",
}

EXPECTED_METRICS = {"performance", "efficiency"}


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
    if cols != REQUIRED_COLUMNS:
        fail(f"{path}: expected columns {sorted(REQUIRED_COLUMNS)}, got {list(df.columns)}")

    tests = set(df["test"].astype(str))
    if tests != EXPECTED_TESTS:
        fail(f"{path}: unexpected test set {sorted(tests)}")

    metrics = set(df["metric"].astype(str))
    if metrics != EXPECTED_METRICS:
        fail(f"{path}: unexpected metric set {sorted(metrics)}")

    for col in ["a100", "rtx5000", "ratio_a100_vs_rtx5000"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            fail(f"{path}: non-numeric values in {col}")
        if (values <= 0).any():
            fail(f"{path}: non-positive values in {col}")


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)

    check_files(compare_dir)
    check_summary(compare_dir / "summary_compare.csv")

    print(f"Comparison artifacts OK: {compare_dir}")


if __name__ == "__main__":
    main()
