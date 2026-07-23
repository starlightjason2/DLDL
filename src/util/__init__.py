"""Utility module: data loading and preprocessing."""

import sys

import matplotlib as mpl
from loguru import logger
from .data_loading import (
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad_norm,
    env_float,
    env_int,
    env_tuple,
)
from .processing import (
    get_use_cores,
    create_binary_labels,
    convert_tensors_to_float,
)

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="INFO",
)


# ----------------------------------------------------------------------------
# Publication-ready matplotlib defaults.
#
# Applied once here because every plotting script (validate.py, graph.py,
# prediction_plots.py, plot_roc_curve.py, ...) imports from ``util`` before it
# touches pyplot, so this runs first. Tuned for print archives: serif type to
# match the LaTeX body (Times), a restrained colorblind-safe palette, light
# gridlines, and vector-friendly output. Individual scripts can still override
# any of these per-figure.
# ----------------------------------------------------------------------------

mpl.rcParams.update(
    {
        # Typography: serif to match the paper body, sized for column figures.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        # Axes / spines: drop the top and right box for a cleaner look.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.titleweight": "bold",
        "axes.axisbelow": True,  # gridlines behind data
        # Grid: faint, unobtrusive.
        "grid.color": "#B0B0B0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        # Ticks: inward, on both major axes.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Lines / markers.
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "scatter.edgecolors": "none",
        # Legend: light frame, no shadow.
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        # Figure / output: white background, tight vector-friendly export.
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,  # embed TrueType (editable text in vector output)
        "ps.fonttype": 42,
    }
)


__all__ = [
    "env_float",
    "env_int",
    "env_tuple",
    "get_length",
    "get_scaled_t_disrupt",
    "get_means",
    "load_and_pad_norm",
    "get_use_cores",
    "create_binary_labels",
    "convert_tensors_to_float",
]
