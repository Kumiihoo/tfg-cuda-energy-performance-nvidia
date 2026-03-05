import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GEMM benchmark results")
    parser.add_argument("--input", default="results/a100/baseline/gemm.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/a100/baseline/gemm.png", help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    plt.figure()
    for tf32, g in df.groupby("tf32"):
        g = g.sort_values("N")
        plt.plot(g["N"], g["GFLOPs"], marker="o", label=f"tf32={tf32}")

    plt.xscale("log", base=2)
    plt.xlabel("Matrix size N")
    plt.ylabel("Throughput (GFLOP/s)")
    plt.title("cuBLAS SGEMM throughput (FP32) with/without TF32")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

