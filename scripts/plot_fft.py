import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cuFFT benchmark results")
    parser.add_argument("--input", default="results/a100/baseline/fft.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/a100/baseline/fft.png", help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input).sort_values("n")

    plt.figure()
    plt.plot(df["n"], df["MSamples_per_s"], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("FFT size N")
    plt.ylabel("Throughput (MSamples/s)")
    plt.title("cuFFT 1D batched C2C throughput")
    plt.grid(True, which="both")
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
