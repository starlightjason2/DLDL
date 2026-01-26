"""Script to visualize training results from train.py output."""

import argparse
import os
import sys

from loguru import logger

# Set matplotlib backend to PyQt6 for interactive display
import matplotlib

matplotlib.use("QtAgg")  # QtAgg uses PyQt6 if available, falls back to PyQt5
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.ticker as ticker
import pandas as pd

from constants import GRAPH_DIR, JOB_ID, PROG_DIR


def _plot_and_configure(
    ax: "plt.Axes",
    epochs: pd.Series,
    series_configs: list[dict],
    ylabel: str,
    title: str,
    ylim: list[float] | None = None,
) -> None:
    """Plot series and configure axis."""
    for series in series_configs:
        kwargs = {
            "label": series["label"],
            "linewidth": 2,
            **({"color": series["color"]} if "color" in series else {}),
        }
        ax.plot(epochs, series["data"], **kwargs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, which="major", alpha=0.4, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.5, linestyle="--")
    ax.minorticks_on()


def plot_training_log(
    csv_path: str, output_path: str | None = None, show_plot: bool = False
) -> None:
    """Plot training metrics from a training log CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"Training log file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Print best performance scores
    best_epoch = df.loc[df["validation_loss"].idxmin()]
    logger.info("=" * 60)
    logger.info("Best Performance (Lowest Validation Loss):")
    logger.info(f"  Epoch: {int(best_epoch['epoch'])}")
    logger.info(f"  Training Loss: {best_epoch['training_loss']:.6f}")
    logger.info(f"  Validation Loss: {best_epoch['validation_loss']:.6f}")
    logger.info(f"  Accuracy: {best_epoch['Validation Accuracy']:.6f}")
    logger.info(f"  Precision: {best_epoch['Validation Precision']:.6f}")
    logger.info(f"  Recall: {best_epoch['Validation Recall']:.6f}")
    logger.info(f"  F1 Score: {best_epoch['Validation F1 Score']:.6f}")
    
    best_accuracy_epoch = df.loc[df["Validation Accuracy"].idxmax()]
    logger.info("-" * 60)
    logger.info("Best Accuracy:")
    logger.info(f"  Epoch: {int(best_accuracy_epoch['epoch'])}")
    logger.info(f"  Accuracy: {best_accuracy_epoch['Validation Accuracy']:.6f}")
    logger.info(f"  Validation Loss: {best_accuracy_epoch['validation_loss']:.6f}")
    logger.info("=" * 60)

    # Define plot configurations: (position, series_configs, ylabel, title, ylim)
    plot_configs = [
        (
            (0, 0),
            [
                {"data": df["training_loss"], "label": "Training Loss"},
                {"data": df["validation_loss"], "label": "Validation Loss"},
            ],
            "Loss",
            "Training and Validation Loss",
            None,
        ),
        (
            (0, 1),
            [
                {
                    "data": df["Validation Accuracy"],
                    "label": "Accuracy",
                    "color": "green",
                }
            ],
            "Accuracy",
            "Validation Accuracy",
            [0, 1],
        ),
        (
            (1, 0),
            [
                {
                    "data": df["Validation Precision"],
                    "label": "Precision",
                    "color": "orange",
                },
                {"data": df["Validation Recall"], "label": "Recall", "color": "purple"},
            ],
            "Score",
            "Validation Precision and Recall",
            [0, 1],
        ),
        (
            (1, 1),
            [{"data": df["Validation F1 Score"], "label": "F1 Score", "color": "red"}],
            "F1 Score",
            "Validation F1 Score",
            [0, 1],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
    for position, series_configs, ylabel, title, ylim in plot_configs:
        _plot_and_configure(axes[position], df["epoch"], series_configs, ylabel, title, ylim)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved graph to {output_path}")

    if show_plot:
        try:
            if plt.get_backend().lower() in ["agg", "pdf", "svg", "ps"]:
                logger.warning(
                    "Interactive display not available. Plot will be saved but not displayed."
                )
                if not output_path:
                    output_path = os.path.join(
                        GRAPH_DIR, f"{JOB_ID}_training_log_plot.png"
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    logger.info(f"Saved graph to {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.warning(
                f"Could not display plot interactively: {e}. Plot will be saved but not displayed."
            )
            if not output_path:
                output_path = os.path.join(GRAPH_DIR, f"{JOB_ID}_training_log_plot.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved graph to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize training results from train.py output"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=f"Path to training log CSV file (default: {PROG_DIR}/{JOB_ID}_training_log.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Path to save the plot image (default: {GRAPH_DIR}/{JOB_ID}_training_log_plot.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (default: False)",
    )

    args = parser.parse_args()

    csv_path = args.csv or os.path.join(PROG_DIR, f"{JOB_ID}_training_log.csv")
    output_path = args.output or (
        os.path.join(GRAPH_DIR, f"{JOB_ID}_training_log_plot.png")
        if not args.show
        else None
    )
    logger.info(f"Loading training log from: {csv_path}")
    if output_path:
        logger.info(f"Output will be saved to: {output_path}")
    plot_training_log(csv_path=csv_path, output_path=output_path, show_plot=args.show)
