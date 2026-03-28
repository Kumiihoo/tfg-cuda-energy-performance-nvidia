from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from perf_targets import load_perf_targets


POWER_COLS = ["ts", "pstate", "power", "sm", "mem", "temp"]
META_REQUIRED_COLS = {
    "case_key",
    "params",
    "perf_unit",
    "avg_perf",
    "target_duration_ms",
    "measured_work_ms",
    "wall_ms",
    "case_repeats",
    "run_start_utc",
    "run_end_utc",
}
META_LOCAL_COLS = {
    "run_start_local",
    "run_end_local",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute efficiency summary from benchmark-delimited stable power windows")
    parser.add_argument("--perf-dir", default="results/a100/baseline", help="Directory with benchmark CSVs")
    parser.add_argument("--power-dir", default="results/a100/energy", help="Directory with power_*_long.csv files")
    parser.add_argument(
        "--output",
        default="results/a100/energy/efficiency_active_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--stable-window-trim",
        type=float,
        default=0.15,
        help="Fraction trimmed from start and end of the benchmark-delimited run window (default: 0.15)",
    )
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=20,
        help="Minimum samples required after trim before falling back to the full run window (default: 20)",
    )
    parser.add_argument(
        "--active-ratio",
        type=float,
        default=0.70,
        help="Diagnostic SM threshold ratio wrt max observed SM clock in the stable window (default: 0.70)",
    )
    parser.add_argument(
        "--active-floor-mhz",
        type=float,
        default=300.0,
        help="Diagnostic absolute lower bound for the SM threshold in MHz (default: 300)",
    )
    return parser.parse_args()


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(" W", "", regex=False)
        .str.replace(" MHz", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_power_timestamps(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed


def _naive_local_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None)
    return ts


def _local_naive_from_utc(value: str) -> pd.Timestamp:
    ts_utc = pd.to_datetime(value, utc=True)
    local_tz = datetime.now().astimezone().tzinfo
    return ts_utc.tz_convert(local_tz).tz_localize(None)


def load_power_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing power log: {path}")

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
    df = df.dropna(subset=["ts", "power", "sm", "mem"]).copy()

    if df.empty:
        raise ValueError(f"No valid rows after cleanup in {path}")

    df["ts_local"] = _parse_power_timestamps(df["ts"])
    df["power_w"] = _to_numeric(df["power"])
    df["sm_mhz"] = _to_numeric(df["sm"])
    df = df.dropna(subset=["ts_local", "power_w", "sm_mhz"]).copy()
    df = df.sort_values("ts_local").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No numeric/timestamp rows in {path}")

    return df


def load_energy_meta(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing energy metadata: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in energy metadata: {path}")

    missing = META_REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    row = df.iloc[0]
    meta = {col: row[col] for col in META_REQUIRED_COLS}
    meta["avg_perf"] = float(meta["avg_perf"])
    meta["target_duration_ms"] = float(meta["target_duration_ms"])
    meta["measured_work_ms"] = float(meta["measured_work_ms"])
    meta["wall_ms"] = float(meta["wall_ms"])
    meta["case_repeats"] = int(float(meta["case_repeats"]))

    if META_LOCAL_COLS.issubset(df.columns):
        meta["run_start_local"] = _naive_local_timestamp(str(row["run_start_local"]))
        meta["run_end_local"] = _naive_local_timestamp(str(row["run_end_local"]))
    else:
        meta["run_start_local"] = _local_naive_from_utc(str(meta["run_start_utc"]))
        meta["run_end_local"] = _local_naive_from_utc(str(meta["run_end_utc"]))
    return meta


def clip_to_run_window(df: pd.DataFrame, meta: dict[str, object]) -> tuple[pd.DataFrame, bool]:
    start_local = meta["run_start_local"]
    end_local = meta["run_end_local"]
    clipped = df[(df["ts_local"] >= start_local) & (df["ts_local"] <= end_local)].copy()
    if not clipped.empty:
        return clipped.reset_index(drop=True), False

    raise ValueError(
        "No logger samples overlap the benchmark interval "
        f"[{meta['run_start_utc']}, {meta['run_end_utc']}]"
    )


def select_stable_window(df: pd.DataFrame, trim_ratio: float, min_window_samples: int) -> tuple[pd.DataFrame, bool, float]:
    if df.empty:
        raise ValueError("Cannot select a stable window from an empty frame")

    requested_trim = max(trim_ratio, 0.0)
    trim_each = int(len(df) * requested_trim)
    max_trim = max((len(df) - min_window_samples) // 2, 0)
    trim_each = min(trim_each, max_trim)

    if trim_each > 0:
        trimmed = df.iloc[trim_each : len(df) - trim_each].copy()
    else:
        trimmed = df.copy()

    fallback = len(trimmed) < min_window_samples
    if fallback:
        trimmed = df.copy()
        applied_trim = 0.0
    else:
        applied_trim = trim_each / len(df)

    return trimmed.reset_index(drop=True), fallback, float(applied_trim)


def duration_ms(df: pd.DataFrame) -> float:
    if len(df) <= 1:
        return 0.0
    delta = df["ts_local"].iloc[-1] - df["ts_local"].iloc[0]
    return float(delta.total_seconds() * 1000.0)


def diagnostic_sm_stats(df: pd.DataFrame, active_ratio: float, active_floor_mhz: float) -> dict[str, float]:
    peak_sm = float(df["sm_mhz"].max())
    threshold = max(active_floor_mhz, peak_sm * active_ratio)
    active_pct = float((df["sm_mhz"] >= threshold).mean() * 100.0)
    return {
        "threshold_mhz": float(threshold),
        "active_pct": active_pct,
    }


def main() -> None:
    args = parse_args()

    perf_dir = Path(args.perf_dir)
    power_dir = Path(args.power_dir)
    targets = load_perf_targets(perf_dir)

    rows: list[tuple[object, ...]] = []
    for target in targets:
        power_path = power_dir / target.power_log_name
        meta_path = power_dir / target.meta_log_name
        power_df = load_power_log(power_path)
        meta = load_energy_meta(meta_path)
        meta_case_key = str(meta["case_key"]).strip()
        meta_perf_unit = str(meta["perf_unit"]).strip()

        if meta_perf_unit != target.perf_unit:
            raise ValueError(
                f"{meta_path}: perf_unit='{meta_perf_unit}' does not match target perf unit '{target.perf_unit}'"
            )
        if not meta_case_key:
            raise ValueError(f"{meta_path}: missing non-empty case_key for target '{target.case_key}'")
        if meta_case_key != target.case_key:
            raise ValueError(
                f"{meta_path}: case_key='{meta_case_key}' does not match expected target case '{target.case_key}'"
            )

        run_window, fallback_run_window = clip_to_run_window(power_df, meta)
        stable_window, fallback_window, applied_trim = select_stable_window(
            run_window, args.stable_window_trim, args.min_window_samples
        )

        active_power_w = float(stable_window["power_w"].mean())
        if active_power_w <= 0.0:
            raise ValueError(f"{power_path}: non-positive active power computed for {target.test}")

        sm_diag = diagnostic_sm_stats(stable_window, args.active_ratio, args.active_floor_mhz)
        energy_perf = float(meta["avg_perf"])
        baseline_perf = float(target.perf)
        if energy_perf <= 0.0:
            raise ValueError(f"{meta_path}: non-positive avg_perf for {target.test}")

        perf_delta_pct = ((energy_perf - baseline_perf) / baseline_perf) * 100.0 if baseline_perf else 0.0

        rows.append(
            (
                target.test,
                target.perf_unit,
                energy_perf,
                active_power_w,
                energy_perf / active_power_w,
                baseline_perf,
                meta["wall_ms"],
                duration_ms(stable_window),
                int(len(stable_window)),
                applied_trim,
                fallback_window,
                float(stable_window["sm_mhz"].mean()),
                float(stable_window["sm_mhz"].max()),
                int(len(run_window)),
                meta["measured_work_ms"],
                meta["target_duration_ms"],
                meta["case_repeats"],
                perf_delta_pct,
                sm_diag["threshold_mhz"],
                sm_diag["active_pct"],
                fallback_run_window,
                meta_case_key,
                target.power_log_name,
                target.meta_log_name,
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
        ],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
