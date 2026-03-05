import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bandwidth benchmark results")
    parser.add_argument("--input", default="results/a100/baseline/bw.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/a100/baseline/bw.png", help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    df["MB"] = df["bytes"] / (1024 * 1024)

    plt.figure()
    plt.plot(df["MB"], df["GBs"], marker="o")
    plt.xscale("log")
    plt.xlabel("Size (MB, log)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Global memory copy bandwidth")
    plt.grid(True, which="both")
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

