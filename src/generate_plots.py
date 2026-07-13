"""Scatter and histogram plots of predicted vs true disruption times from ``best_model/predictions.csv``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from util.best_model import best_model_dir, load_best_model_env


def generate_histogram(df: pd.DataFrame) -> None:
    diff = (df["true_time"] - df["predicted_time"]) * 1e3

    print(
        f"Disruption time error (milliseconds, n={len(diff)}): "
        f"mean={diff.mean():.3f}, variance={diff.var():.3f}, stddev={diff.std():.3f}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff, bins=50, edgecolor="black", alpha=0.85, range=(-7.5, 7.5))
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Difference from real disruption time (milliseconds)")
    ax.set_ylabel("Count (# shots)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "disruption_time_diff.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"Wrote {out_path}")


def generate_scatter_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["predicted_time"], df["true_time"], alpha=0.85)
    lo = min(df["predicted_time"].min(), df["true_time"].min())
    hi = max(df["predicted_time"].max(), df["true_time"].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Predicted disruption time (s)")
    ax.set_ylabel("True disruption time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / "predictions_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    load_best_model_env()
    model_dir = best_model_dir()
    csv_path = model_dir / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run validate.py first to generate predictions."
        )
    df = pd.read_csv(csv_path)

    generate_scatter_plot(df)
    generate_histogram(df)
