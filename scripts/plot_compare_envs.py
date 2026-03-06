from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TEST_ORDER = [
    "BW plateau",
    "Compute FP32 peak",
    "Compute FP64 peak",
    "GEMM TF32=0 max",
    "GEMM TF32=1 max",
]

TEST_LABEL = {
    "BW plateau": "BW",
    "Compute FP32 peak": "Compute FP32",
    "Compute FP64 peak": "Compute FP64",
    "GEMM TF32=0 max": "GEMM TF32=0",
    "GEMM TF32=1 max": "GEMM TF32=1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare A100 vs RTX5000 benchmark and efficiency results")
    parser.add_argument("--a100-root", default="results/a100", help="Root folder for A100 results")
    parser.add_argument("--rtx5000-root", default="results/rtx5000", help="Root folder for RTX5000 results")
    parser.add_argument("--output-dir", default="results/compare", help="Output folder for comparison artifacts")
    return parser.parse_args()


def fail(msg: str) -> None:
    raise ValueError(msg)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        fail(f"No rows in {path}")
    return df


def require_columns(path: Path, df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols.difference(df.columns)
    if missing:
        fail(f"{path}: missing columns {sorted(missing)}")


def as_float(value: object, ctx: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive path
        fail(f"Cannot parse float for {ctx}: {value} ({exc})")
    if not pd.notna(out):
        fail(f"NaN value for {ctx}")
    return out


def max_or_fail(series: pd.Series, ctx: str) -> float:
    if series.empty:
        fail(f"No rows for {ctx}")
    val = as_float(series.max(), ctx)
    if val <= 0:
        fail(f"Non-positive value for {ctx}: {val}")
    return val


def load_performance(env_root: Path) -> dict[str, tuple[str, float]]:
    baseline = env_root / "baseline"

    bw_path = baseline / "bw.csv"
    compute_path = baseline / "compute.csv"
    gemm_path = baseline / "gemm.csv"

    bw = read_csv(bw_path)
    compute = read_csv(compute_path)
    gemm = read_csv(gemm_path)

    require_columns(bw_path, bw, {"bytes", "iters", "block", "GBs"})
    require_columns(compute_path, compute, {"dtype", "block", "grid", "iters", "GFLOPs"})
    require_columns(gemm_path, gemm, {"N", "iters", "tf32", "GFLOPs"})

    c = compute.copy()
    c["dtype"] = c["dtype"].astype(str).str.strip().str.lower()

    g = gemm.copy()
    g["tf32"] = pd.to_numeric(g["tf32"], errors="coerce")

    perf = {
        "BW plateau": ("GB/s", max_or_fail(pd.to_numeric(bw["GBs"], errors="coerce"), "BW plateau")),
        "Compute FP32 peak": (
            "GFLOP/s",
            max_or_fail(pd.to_numeric(c.loc[c["dtype"] == "fp32", "GFLOPs"], errors="coerce"), "Compute FP32 peak"),
        ),
        "Compute FP64 peak": (
            "GFLOP/s",
            max_or_fail(pd.to_numeric(c.loc[c["dtype"] == "fp64", "GFLOPs"], errors="coerce"), "Compute FP64 peak"),
        ),
        "GEMM TF32=0 max": (
            "GFLOP/s",
            max_or_fail(pd.to_numeric(g.loc[g["tf32"] == 0, "GFLOPs"], errors="coerce"), "GEMM TF32=0 max"),
        ),
        "GEMM TF32=1 max": (
            "GFLOP/s",
            max_or_fail(pd.to_numeric(g.loc[g["tf32"] == 1, "GFLOPs"], errors="coerce"), "GEMM TF32=1 max"),
        ),
    }
    return perf


def load_efficiency(env_root: Path) -> dict[str, float]:
    path = env_root / "energy" / "efficiency_active_summary.csv"
    df = read_csv(path)
    require_columns(path, df, {"Test", "Efficiency (unit/W)"})

    out: dict[str, float] = {}
    for test in TEST_ORDER:
        row = df.loc[df["Test"].astype(str).str.strip() == test]
        if row.empty:
            fail(f"{path}: missing test row '{test}'")
        val = as_float(row.iloc[0]["Efficiency (unit/W)"], f"{path}:{test}:Efficiency")
        if val <= 0:
            fail(f"{path}: non-positive efficiency for '{test}': {val}")
        out[test] = val
    return out


def env_frame(env_name: str, env_root: Path) -> pd.DataFrame:
    perf = load_performance(env_root)
    eff = load_efficiency(env_root)

    rows: list[dict[str, object]] = []
    for test in TEST_ORDER:
        p_unit, p_val = perf[test]
        rows.append({"test": test, "metric": "performance", "unit": p_unit, "value": p_val, "env": env_name})
        rows.append({"test": test, "metric": "efficiency", "unit": "unit/W", "value": eff[test], "env": env_name})
    return pd.DataFrame(rows)


def build_compare_table(a100: pd.DataFrame, rtx: pd.DataFrame) -> pd.DataFrame:
    left = a100.rename(columns={"value": "a100"}).drop(columns=["env"])
    right = rtx.rename(columns={"value": "rtx5000"}).drop(columns=["env"])

    merged = pd.merge(left, right, on=["test", "metric", "unit"], how="inner")
    if merged.empty:
        fail("No overlap between A100 and RTX5000 metrics")

    merged["ratio_a100_vs_rtx5000"] = merged["a100"] / merged["rtx5000"]
    merged["delta_percent"] = (merged["ratio_a100_vs_rtx5000"] - 1.0) * 100.0

    merged["test"] = pd.Categorical(merged["test"], TEST_ORDER, ordered=True)
    metric_order = pd.CategoricalDtype(["performance", "efficiency"], ordered=True)
    merged["metric"] = merged["metric"].astype(metric_order)
    merged = merged.sort_values(["metric", "test"]).reset_index(drop=True)
    merged["test"] = merged["test"].astype(str)
    merged["metric"] = merged["metric"].astype(str)
    return merged


def plot_grouped(compare_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    df = compare_df[compare_df["metric"] == metric].copy()
    if df.empty:
        fail(f"No rows to plot for metric={metric}")

    labels = [TEST_LABEL.get(t, t) for t in df["test"].tolist()]
    x = list(range(len(labels)))
    w = 0.38

    plt.figure(figsize=(11, 4.8))
    plt.bar([i - w / 2 for i in x], df["a100"], width=w, label="A100")
    plt.bar([i + w / 2 for i in x], df["rtx5000"], width=w, label="RTX5000")

    plt.xticks(x, labels, rotation=20, ha="right")
    if metric == "performance":
        plt.ylabel("Performance (native units)")
        plt.title("A100 vs RTX5000: Absolute Performance")
    else:
        plt.ylabel("Efficiency (unit/W)")
        plt.title("A100 vs RTX5000: Energy Efficiency")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_ratio(compare_df: pd.DataFrame, out_path: Path) -> None:
    df = compare_df.copy()
    labels = [f"{TEST_LABEL.get(t, t)}\n({'perf' if m == 'performance' else 'eff'})" for t, m in zip(df["test"], df["metric"])]
    colors = ["tab:blue" if m == "performance" else "tab:green" for m in df["metric"]]

    plt.figure(figsize=(12, 5.4))
    plt.bar(range(len(df)), df["ratio_a100_vs_rtx5000"], color=colors)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(range(len(df)), labels, rotation=20, ha="right")
    plt.ylabel("Ratio (A100 / RTX5000)")
    plt.title("A100 vs RTX5000: Relative Gain")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()

    a100_root = Path(args.a100_root)
    rtx_root = Path(args.rtx5000_root)
    output_dir = Path(args.output_dir)

    a100_df = env_frame("a100", a100_root)
    rtx_df = env_frame("rtx5000", rtx_root)
    compare_df = build_compare_table(a100_df, rtx_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_compare.csv"
    compare_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    plot_grouped(compare_df, "performance", output_dir / "perf_absolute_compare.png")
    plot_grouped(compare_df, "efficiency", output_dir / "efficiency_compare.png")
    plot_ratio(compare_df, output_dir / "speedup_a100_vs_rtx5000.png")


if __name__ == "__main__":
    main()
