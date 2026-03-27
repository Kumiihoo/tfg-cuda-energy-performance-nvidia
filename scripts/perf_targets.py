from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TEST_ORDER = [
    "BW peak",
    "BW sustained",
    "Compute FP32 peak",
    "Compute FP64 peak",
    "GEMM TF32=0 max",
    "GEMM TF32=1 max",
]

TEST_LABEL = {
    "BW peak": "BW peak",
    "BW sustained": "BW sustained",
    "Compute FP32 peak": "Compute FP32",
    "Compute FP64 peak": "Compute FP64",
    "GEMM TF32=0 max": "GEMM TF32=0",
    "GEMM TF32=1 max": "GEMM TF32=1",
}


@dataclass(frozen=True)
class PerfTarget:
    case_key: str
    test: str
    perf_unit: str
    perf: float
    power_log_name: str
    meta_log_name: str
    bench_args: tuple[str, ...]


def _meta_name_from_power_log(power_log_name: str) -> str:
    if power_log_name.endswith(".csv"):
        return power_log_name[:-4] + "_meta.csv"
    return power_log_name + "_meta.csv"


def choose_perf_file(perf_dir: Path, stem: str) -> Path:
    candidates = [
        perf_dir / f"{stem}.csv",
        perf_dir / f"{stem}_baseline.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing performance CSV for '{stem}' in {perf_dir}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    return df


def _require_columns(path: Path, df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _numeric_frame(df: pd.DataFrame, cols: list[str], path: Path) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=cols).copy()
    if out.empty:
        raise ValueError(f"{path}: no valid numeric rows for columns {cols}")
    return out


def load_perf_targets(perf_dir: Path) -> list[PerfTarget]:
    bw_path = choose_perf_file(perf_dir, "bw")
    compute_path = choose_perf_file(perf_dir, "compute")
    gemm_path = choose_perf_file(perf_dir, "gemm")

    bw = _read_csv(bw_path)
    compute = _read_csv(compute_path)
    gemm = _read_csv(gemm_path)

    _require_columns(bw_path, bw, {"bytes", "iters", "block", "GBs"})
    _require_columns(compute_path, compute, {"dtype", "block", "grid", "iters", "GFLOPs"})
    _require_columns(gemm_path, gemm, {"N", "iters", "tf32", "GFLOPs"})

    bw_num = _numeric_frame(bw, ["bytes", "iters", "block", "GBs"], bw_path)
    bw_peak_row = bw_num.loc[bw_num["GBs"].idxmax()]
    bw_sustained_row = bw_num.iloc[-1]

    compute_num = _numeric_frame(compute, ["block", "grid", "iters", "GFLOPs"], compute_path)
    compute_num["dtype"] = compute_num["dtype"].astype(str).str.strip().str.lower()

    gemm_num = _numeric_frame(gemm, ["N", "iters", "tf32", "GFLOPs"], gemm_path)
    gemm_num["tf32"] = gemm_num["tf32"].astype(int)

    targets = [
        PerfTarget(
            case_key="bw_peak",
            test="BW peak",
            perf_unit="GB/s",
            perf=float(bw_peak_row["GBs"]),
            power_log_name="power_bw_peak_long.csv",
            meta_log_name=_meta_name_from_power_log("power_bw_peak_long.csv"),
            bench_args=(
                "--mode",
                "bw",
                "--bw-bytes",
                str(int(bw_peak_row["bytes"])),
                "--bw-iters",
                str(int(bw_peak_row["iters"])),
                "--bw-block",
                str(int(bw_peak_row["block"])),
            ),
        ),
        PerfTarget(
            case_key="bw_sustained",
            test="BW sustained",
            perf_unit="GB/s",
            perf=float(bw_sustained_row["GBs"]),
            power_log_name="power_bw_sustained_long.csv",
            meta_log_name=_meta_name_from_power_log("power_bw_sustained_long.csv"),
            bench_args=(
                "--mode",
                "bw",
                "--bw-bytes",
                str(int(bw_sustained_row["bytes"])),
                "--bw-iters",
                str(int(bw_sustained_row["iters"])),
                "--bw-block",
                str(int(bw_sustained_row["block"])),
            ),
        ),
    ]

    for dtype, case_key, test, power_log_name in [
        ("fp32", "compute_fp32_peak", "Compute FP32 peak", "power_compute_fp32_peak_long.csv"),
        ("fp64", "compute_fp64_peak", "Compute FP64 peak", "power_compute_fp64_peak_long.csv"),
    ]:
        subset = compute_num[compute_num["dtype"] == dtype]
        if subset.empty:
            raise ValueError(f"{compute_path}: no rows for dtype '{dtype}'")
        row = subset.loc[subset["GFLOPs"].idxmax()]
        targets.append(
            PerfTarget(
                case_key=case_key,
                test=test,
                perf_unit="GFLOP/s",
                perf=float(row["GFLOPs"]),
                power_log_name=power_log_name,
                meta_log_name=_meta_name_from_power_log(power_log_name),
                bench_args=(
                    "--mode",
                    "compute",
                    "--compute-dtype",
                    dtype,
                    "--compute-block",
                    str(int(row["block"])),
                    "--compute-grid",
                    str(int(row["grid"])),
                    "--compute-iters",
                    str(int(row["iters"])),
                ),
            )
        )

    for tf32, case_key, test, power_log_name in [
        (0, "gemm_tf32_0_max", "GEMM TF32=0 max", "power_gemm_tf32_0_max_long.csv"),
        (1, "gemm_tf32_1_max", "GEMM TF32=1 max", "power_gemm_tf32_1_max_long.csv"),
    ]:
        subset = gemm_num[gemm_num["tf32"] == tf32]
        if subset.empty:
            raise ValueError(f"{gemm_path}: no rows for tf32={tf32}")
        row = subset.loc[subset["GFLOPs"].idxmax()]
        targets.append(
            PerfTarget(
                case_key=case_key,
                test=test,
                perf_unit="GFLOP/s",
                perf=float(row["GFLOPs"]),
                power_log_name=power_log_name,
                meta_log_name=_meta_name_from_power_log(power_log_name),
                bench_args=(
                    "--mode",
                    "gemm",
                    "--gemm-n",
                    str(int(row["N"])),
                    "--gemm-iters",
                    str(int(row["iters"])),
                    "--gemm-tf32",
                    str(tf32),
                ),
            )
        )

    order = {test: idx for idx, test in enumerate(TEST_ORDER)}
    return sorted(targets, key=lambda target: order[target.test])


def load_target_map(perf_dir: Path) -> dict[str, PerfTarget]:
    return {target.case_key: target for target in load_perf_targets(perf_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit benchmark target metadata from baseline CSV files")
    parser.add_argument("--perf-dir", default="results/a100/baseline", help="Directory with bw/compute/gemm CSVs")
    parser.add_argument("--case", required=True, help="Target case key")
    parser.add_argument(
        "--field",
        required=True,
        choices=["args", "perf", "perf-unit", "power-log", "meta-log", "test"],
        help="Target field to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_target_map(Path(args.perf_dir))
    if args.case not in targets:
        raise ValueError(f"Unknown case '{args.case}'. Available cases: {sorted(targets)}")

    target = targets[args.case]
    if args.field == "args":
        print(" ".join(target.bench_args))
    elif args.field == "perf":
        print(target.perf)
    elif args.field == "perf-unit":
        print(target.perf_unit)
    elif args.field == "power-log":
        print(target.power_log_name)
    elif args.field == "meta-log":
        print(target.meta_log_name)
    elif args.field == "test":
        print(target.test)


if __name__ == "__main__":
    main()
