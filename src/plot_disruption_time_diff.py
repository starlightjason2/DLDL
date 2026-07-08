"""Histogram of predicted minus true disruption time from ``best_model/predictions.csv``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from util.best_model import best_model_dir, load_best_model_env


def main() -> None:
    load_best_model_env()
    model_dir = best_model_dir()
    csv_path = model_dir / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run validate.py first to generate predictions."
        )

    df = pd.read_csv(csv_path)
    diff = df["predicted_time"] - df["true_time"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff, bins=30, edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted - true disruption time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Disruption time prediction error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "disruption_time_diff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
