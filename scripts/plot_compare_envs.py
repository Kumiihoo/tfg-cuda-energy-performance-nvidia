from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from perf_targets import TEST_LABEL, TEST_ORDER, load_perf_targets

PERF_TEST_ORDER = TEST_ORDER + ["FFT C2C max"]
EXTRA_TEST_LABEL = {"FFT C2C max": "FFT C2C"}


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


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_run_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def extract_nvcc_version(text: str) -> str:
    m = re.search(r"release\s+([0-9.]+)", text)
    return m.group(1) if m else ""


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def load_env_metadata(env_name: str, env_root: Path) -> dict[str, str]:
    env_dir = env_root / "env"
    cfg = parse_run_config(env_dir / "run_config.txt")
    nvcc_text = read_text_if_exists(env_dir / "env_nvcc.txt")
    python_text = read_text_if_exists(env_dir / "env_python.txt")

    return {
        "environment": env_name,
        "host": cfg.get("host", "unknown"),
        "gpu_name": cfg.get("gpu_name", "unknown"),
        "git_commit": cfg.get("git_commit", "unknown"),
        "driver_version": cfg.get("driver_version", "unknown"),
        "cuda_driver_version": cfg.get("cuda_driver_version", "unknown"),
        "nvcc_version": cfg.get("nvcc_version", extract_nvcc_version(nvcc_text) or "unknown"),
        "python_version": cfg.get("python_version", first_nonempty_line(python_text) or "unknown"),
        "sample_ms": cfg.get("sample_ms", "unknown"),
        "energy_duration_ms": cfg.get("energy_duration_ms", "unknown"),
        "stable_window_trim": cfg.get("stable_window_trim", "unknown"),
        "power_telemetry_source": cfg.get("power_telemetry_source", "nvidia-smi"),
        "power_scope": cfg.get("power_scope", "gpu_board"),
        "node_power_not_measured": cfg.get("node_power_not_measured", "1"),
        "activity_definition": cfg.get("activity_definition", "unknown"),
    }


def build_env_compare(a100_meta: dict[str, str], rtx_meta: dict[str, str]) -> pd.DataFrame:
    field_specs = [
        ("identity", "environment", False),
        ("identity", "host", False),
        ("identity", "gpu_name", False),
        ("reproducibility", "git_commit", True),
        ("software", "driver_version", True),
        ("software", "cuda_driver_version", True),
        ("software", "nvcc_version", True),
        ("software", "python_version", True),
        ("measurement", "sample_ms", True),
        ("measurement", "energy_duration_ms", True),
        ("measurement", "stable_window_trim", True),
        ("measurement", "power_telemetry_source", True),
        ("measurement", "power_scope", True),
        ("measurement", "node_power_not_measured", True),
        ("measurement", "activity_definition", True),
    ]

    rows: list[dict[str, object]] = []
    for category, field, expected_match in field_specs:
        a100_value = a100_meta.get(field, "unknown")
        rtx_value = rtx_meta.get(field, "unknown")
        match = a100_value == rtx_value
        missing = a100_value in {"", "unknown"} or rtx_value in {"", "unknown"}
        status = "warning" if expected_match and (not match or missing) else "ok"
        if not expected_match:
            status = "info"
        rows.append(
            {
                "category": category,
                "field": field,
                "a100": a100_value,
                "rtx5000": rtx_value,
                "match": match,
                "expected_match": expected_match,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def write_methodology_notes(compare_meta: pd.DataFrame, out_path: Path) -> None:
    warnings = compare_meta[compare_meta["status"] == "warning"]
    lines = [
        "Methodology notes for cross-environment comparison",
        "",
        "Warnings:",
    ]

    if warnings.empty:
        lines.append("- No mismatches detected in strict-match metadata fields.")
    else:
        for _, row in warnings.iterrows():
            lines.append(
                f"- {row['field']}: A100='{row['a100']}' vs RTX5000='{row['rtx5000']}'"
            )

    lines.extend(
        [
            "",
            "Measurement scope:",
            "- Power telemetry comes from nvidia-smi.",
            "- The reported power scope is GPU-board only; whole-node/system power is not measured by this project.",
            "- Energy efficiency is computed from the benchmark-delimited stable window, not from a global SM-clock activity filter.",
            "- Cross-environment comparisons should therefore be interpreted as GPU-centric, not node-level energy efficiency.",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def load_performance(env_root: Path) -> dict[str, tuple[str, float]]:
    perf: dict[str, tuple[str, float]] = {}
    for target in load_perf_targets(env_root / "baseline"):
        perf[target.test] = (target.perf_unit, target.perf)

    fft_path = env_root / "baseline" / "fft.csv"
    if fft_path.exists():
        fft = read_csv(fft_path)
        require_columns(fft_path, fft, {"n", "batch", "iters", "MSamples_per_s"})
        fft["MSamples_per_s"] = pd.to_numeric(fft["MSamples_per_s"], errors="coerce")
        fft = fft.dropna(subset=["MSamples_per_s"]).copy()
        if not fft.empty:
            row = fft.loc[fft["MSamples_per_s"].idxmax()]
            perf["FFT C2C max"] = ("MSamples/s", as_float(row["MSamples_per_s"], f"{fft_path}:FFT C2C max"))
    return perf


def load_efficiency(env_root: Path) -> dict[str, float]:
    path = env_root / "energy" / "efficiency_active_summary.csv"
    df = read_csv(path)
    require_columns(path, df, {"Test", "Efficiency (unit/W)"})
    known_tests = set(df["Test"].astype(str).str.strip())

    out: dict[str, float] = {}
    for test in TEST_ORDER:
        row = df.loc[df["Test"].astype(str).str.strip() == test]
        if row.empty:
            if test in {"BW peak", "BW sustained"} and "BW plateau" in known_tests:
                fail(
                    f"{path}: legacy BW row 'BW plateau' found. Regenerate the energy summary "
                    "so it includes both 'BW peak' and 'BW sustained'."
                )
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
    for test in PERF_TEST_ORDER:
        if test not in perf:
            continue
        p_unit, p_val = perf[test]
        rows.append({"test": test, "metric": "performance", "unit": p_unit, "value": p_val, "env": env_name})
    for test in TEST_ORDER:
        p_unit, _ = perf[test]
        rows.append({"test": test, "metric": "efficiency", "unit": f"{p_unit}/W", "value": eff[test], "env": env_name})
    return pd.DataFrame(rows)


def build_compare_table(a100: pd.DataFrame, rtx: pd.DataFrame) -> pd.DataFrame:
    left = a100.rename(columns={"value": "a100"}).drop(columns=["env"])
    right = rtx.rename(columns={"value": "rtx5000"}).drop(columns=["env"])

    merged = pd.merge(left, right, on=["test", "metric", "unit"], how="inner")
    if merged.empty:
        fail("No overlap between A100 and RTX5000 metrics")

    merged["ratio_a100_vs_rtx5000"] = merged["a100"] / merged["rtx5000"]
    merged["delta_percent"] = (merged["ratio_a100_vs_rtx5000"] - 1.0) * 100.0

    merged["test"] = pd.Categorical(merged["test"], PERF_TEST_ORDER, ordered=True)
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

    units = list(dict.fromkeys(df["unit"].astype(str).tolist()))
    fig, axes = plt.subplots(len(units), 1, figsize=(11, 4.6 * len(units)), squeeze=False)

    for ax, unit in zip(axes.flat, units):
        sub = df[df["unit"].astype(str) == unit].copy()
        labels = [EXTRA_TEST_LABEL.get(t, TEST_LABEL.get(t, t)) for t in sub["test"].tolist()]
        x = list(range(len(labels)))
        w = 0.38

        ax.bar([i - w / 2 for i in x], sub["a100"], width=w, label="A100")
        ax.bar([i + w / 2 for i in x], sub["rtx5000"], width=w, label="RTX5000")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        if metric == "performance":
            ax.set_title(f"Absolute Performance ({unit})")
        else:
            ax.set_title(f"Energy Efficiency ({unit})")

    if metric == "performance":
        fig.suptitle("A100 vs RTX5000: Absolute Performance", y=0.99)
    else:
        fig.suptitle("A100 vs RTX5000: Energy Efficiency", y=0.99)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ratio(compare_df: pd.DataFrame, out_path: Path) -> None:
    df = compare_df.copy()
    labels = [f"{EXTRA_TEST_LABEL.get(t, TEST_LABEL.get(t, t))}\n({'perf' if m == 'performance' else 'eff'})" for t, m in zip(df["test"], df["metric"])]
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
    compare_meta = build_env_compare(load_env_metadata("a100", a100_root), load_env_metadata("rtx5000", rtx_root))

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_compare.csv"
    compare_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    meta_path = output_dir / "environment_compare.csv"
    compare_meta.to_csv(meta_path, index=False)
    print(f"Saved {meta_path}")

    write_methodology_notes(compare_meta, output_dir / "methodology_notes.txt")

    plot_grouped(compare_df, "performance", output_dir / "perf_absolute_compare.png")
    plot_grouped(compare_df, "efficiency", output_dir / "efficiency_compare.png")
    plot_ratio(compare_df, output_dir / "speedup_a100_vs_rtx5000.png")


if __name__ == "__main__":
    main()
