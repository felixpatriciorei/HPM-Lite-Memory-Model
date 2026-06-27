"""Plot HPM-Lite memory model results.

Usage:
    python scripts/plot_results.py --input results/memory_model_results_current.csv --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seq_len": float(row["seq_len"]),
                    "local_exact": float(row["local_exact"]),
                    "hpm_exact": float(row["hpm_exact"]),
                    "gain": float(row["gain"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/memory_model_results_current.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    rows.sort(key=lambda r: r["seq_len"])

    x = [int(r["seq_len"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, [r["local_exact"] for r in rows], marker="o", label="Local Transformer")
    plt.plot(x, [r["hpm_exact"] for r in rows], marker="o", label="HPM-Lite")
    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])
    plt.xlabel("Sequence length / fact-to-query distance proxy")
    plt.ylabel("Exact answer accuracy")
    plt.title("Exact recall vs. distance")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "exact_recall_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar([str(v) for v in x], [r["gain"] for r in rows])
    plt.xlabel("Sequence length")
    plt.ylabel("Accuracy gain")
    plt.title("HPM-Lite exact-recall gain over local baseline")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(args.out_dir / "exact_gain_by_distance.png", dpi=180)
    plt.close()

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
