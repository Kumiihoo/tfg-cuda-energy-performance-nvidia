import argparse
from pathlib import Path

import pandas as pd
from perf_targets import load_perf_targets


POWER_COLS = ["ts", "pstate", "power", "sm", "mem", "temp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute efficiency summary using active power filtering")
    parser.add_argument("--perf-dir", default="results/a100/baseline", help="Directory with benchmark CSVs")
    parser.add_argument("--power-dir", default="results/a100/energy", help="Directory with power_*_long.csv files")
    parser.add_argument(
        "--output",
        default="results/a100/energy/efficiency_active_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--active-ratio",
        type=float,
        default=0.70,
        help="Active threshold ratio wrt max observed SM clock (default: 0.70)",
    )
    parser.add_argument(
        "--active-floor-mhz",
        type=float,
        default=300.0,
        help="Absolute lower bound for active threshold in MHz (default: 300)",
    )
    return parser.parse_args()

def _to_numeric(series: pd.Series) -> pd.Series:
    # Accept logs with or without units and both decimal separators.
    cleaned = (
        series.astype(str)
        .str.replace(" W", "", regex=False)
        .str.replace(" MHz", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_power_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing power log: {path}")

    # Use the Python engine and skip malformed lines to avoid aborting on sporadic row corruption.
    df = pd.read_csv(
        path,
        header=None,
        names=POWER_COLS,
        engine="python",
        on_bad_lines="skip",
        skipinitialspace=True,
    )

    na_re = r"(?i)n/?a|not supported"
    df = df[~df["power"].astype(str).str.contains(na_re, na=False, regex=True)]
    df = df[~df["sm"].astype(str).str.contains(na_re, na=False, regex=True)]
    df = df[~df["mem"].astype(str).str.contains(na_re, na=False, regex=True)]
    df = df.dropna(subset=["power", "sm", "mem"]).copy()

    if df.empty:
        raise ValueError(f"No valid rows after cleanup in {path}")

    df["power_w"] = _to_numeric(df["power"])
    df["sm_mhz"] = _to_numeric(df["sm"])
    df = df.dropna(subset=["power_w", "sm_mhz"]).copy()

    if df.empty:
        raise ValueError(f"No numeric power/sm rows in {path}")

    return df


def active_power_stats(df: pd.DataFrame, active_ratio: float, active_floor_mhz: float) -> dict:
    peak_sm = float(df["sm_mhz"].max())
    threshold = max(active_floor_mhz, peak_sm * active_ratio)

    active = df[df["sm_mhz"] >= threshold]
    used_fallback = False

    if active.empty:
        active = df
        used_fallback = True

    return {
        "power_w": float(active["power_w"].mean()),
        "threshold_mhz": float(threshold),
        "active_pct": float(len(active) * 100.0 / len(df)),
        "samples": int(len(df)),
        "peak_sm_mhz": peak_sm,
        "fallback_all_samples": used_fallback,
    }


def main() -> None:
    args = parse_args()

    perf_dir = Path(args.perf_dir)
    power_dir = Path(args.power_dir)
    targets = load_perf_targets(perf_dir)

    rows = []
    for target in targets:
        stats = active_power_stats(
            load_power_log(power_dir / target.power_log_name), args.active_ratio, args.active_floor_mhz
        )
        rows.append(
            (
                target.test,
                target.perf_unit,
                target.perf,
                stats["power_w"],
                target.perf / stats["power_w"],
                stats["threshold_mhz"],
                stats["active_pct"],
                stats["samples"],
                stats["peak_sm_mhz"],
                stats["fallback_all_samples"],
            )
        )

    out = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
