import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot compute benchmark results")
    parser.add_argument("--input", default="results/a100/baseline/compute.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/a100/baseline/compute.png", help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    plt.figure()
    for dtype, g in df.groupby("dtype"):
        g = g.sort_values("block")
        plt.plot(g["block"], g["GFLOPs"], marker="o", label=dtype)

    plt.xscale("log", base=2)
    plt.xlabel("Block size (threads)")
    plt.ylabel("Throughput (GFLOP/s)")
    plt.title("Compute throughput vs block size (FMA loop)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

