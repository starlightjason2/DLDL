"""Script to visualize training results from train.py output."""

import argparse
import os
import sys

from loguru import logger

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd

from constants import JOB_ID, PROG_DIR


def _configure_axis(
    ax: "plt.Axes",
    ylabel: str,
    title: str,
    ylim: list[float] | None = None,
) -> None:
    """Configure axis labels, title, legend, and grid.

    Args:
        ax: Matplotlib axis to configure.
        ylabel: Y-axis label.
        title: Plot title.
        ylim: Optional Y-axis limits.
    """
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)


def _plot_series(
    ax: "plt.Axes",
    epochs: pd.Series,
    series_configs: list[dict],
) -> None:
    """Plot multiple series on an axis.

    Args:
        ax: Matplotlib axis to plot on.
        epochs: Epoch data for x-axis.
        series_configs: List of dicts with 'data', 'label', and optional 'color'.
    """
    for series in series_configs:
        plot_kwargs = {"label": series["label"], "linewidth": 2}
        if "color" in series:
            plot_kwargs["color"] = series["color"]
        ax.plot(epochs, series["data"], **plot_kwargs)


def plot_training_log(
    csv_path: str,
    output_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Plot training metrics from a training log CSV file.

    Args:
        csv_path: Path to the training log CSV file.
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot (default: False).
    """
    if not os.path.exists(csv_path):
        logger.error(f"Training log file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

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
            [{"data": df["Validation Accuracy"], "label": "Accuracy", "color": "green"}],
            "Accuracy",
            "Validation Accuracy",
            [0, 1],
        ),
        (
            (1, 0),
            [
                {"data": df["Validation Precision"], "label": "Precision", "color": "orange"},
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

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

    # Plot each configuration
    for position, series_configs, ylabel, title, ylim in plot_configs:
        ax = axes[position]
        _plot_series(ax, df["epoch"], series_configs)
        _configure_axis(ax, ylabel, title, ylim)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training plot saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main() -> None:
    """Main entry point for graph.py script."""
    parser = argparse.ArgumentParser(
        description="Visualize training results from train.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths from constants
  python -m src.graph

  # Specify custom CSV file
  python -m src.graph --csv path/to/training_log.csv

  # Save plot to file
  python -m src.graph --output plot.png

  # Display plot interactively
  python -m src.graph --show
        """,
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
        help="Path to save the plot image (default: don't save)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (default: False)",
    )

    args = parser.parse_args()

    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(PROG_DIR, f"{JOB_ID}_training_log.csv")

    logger.info(f"Loading training log from: {csv_path}")

    # Determine output path
    output_path = args.output
    if output_path is None and not args.show:
        # Default: save to same directory as CSV with .png extension
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_plot.png"

    plot_training_log(
        csv_path=csv_path,
        output_path=output_path,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
