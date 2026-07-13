"""Render a paper-ready architecture diagram of the best trained model."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Prevent macOS OpenMP thread deadlock on CPU inference
if not torch.cuda.is_available():
    torch.set_num_threads(1)

from loguru import logger

from model.cnn import IpCNN
from model.dataset import IpDataset
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env

_REPO = Path(__file__).resolve().parents[1]


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def create_model_diagram(model: IpCNN, fig_path: Path) -> Path:
    """Render an architecture diagram in the Transformer-figure idiom."""
    # Palette echoing the "Attention Is All You Need" figure.
    C_COMPUTE = "#ffd8a8"  # Conv1d / Linear
    C_NORM = "#f5edb3"  # BatchNorm
    C_ACT = "#c8e6c9"  # ReLU / Sigmoid
    C_POOL = "#bcdff1"  # MaxPool
    C_DROP = "#e2e2e2"  # Dropout
    C_IO = "#f6cfd0"  # Input embedding-style box
    C_LINOUT = "#d0d0ec"  # final Linear

    pool_k = model.pool.kernel_size
    convs = [model.conv1, model.conv2, model.conv3, model.conv4]

    # Trace tensor shapes stage-by-stage so the annotations stay correct.
    with torch.no_grad():
        x = torch.zeros(1, 1, model.max_length)
        conv_shapes = []
        for conv in convs:
            x = model.pool(F.relu(conv(x)))
            conv_shapes.append((conv.out_channels, int(x.shape[-1])))
        flat_features = x.numel()

    # Geometry (data units).
    sw, sh, sgap, pad = 2.6, 0.42, 0.1, 0.16
    arrow_len = 0.48
    cont_dx = sw / 2 + pad  # half-width of a container
    col_gap = 3.2  # horizontal space between the two columns
    col_x = [0.0, sw + col_gap]  # x-center of each column

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    def draw_box(cx, y0, h, color, label, w=sw, fs=8.5, bold=False, edge="#333333"):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, y0),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.1,
                edgecolor=edge,
                facecolor=color,
            )
        )
        ax.text(
            cx,
            y0 + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
        )

    def draw_container(cx, y_top, sublayers):
        """Draw a light container with colored sub-layers, first one on top."""
        n = len(sublayers)
        inner_h = n * sh + (n - 1) * sgap
        cont_h = inner_h + 2 * pad
        ax.add_patch(
            FancyBboxPatch(
                (cx - (sw / 2 + pad), y_top - cont_h),
                sw + 2 * pad,
                cont_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="#888888",
                facecolor="#f4f4f4",
            )
        )
        yy = y_top - pad - sh  # bottom edge of the top-most sub-layer
        for label, color in sublayers:
            ax.add_patch(
                FancyBboxPatch(
                    (cx - sw / 2, yy),
                    sw,
                    sh,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    linewidth=1.0,
                    edgecolor="#555555",
                    facecolor=color,
                )
            )
            ax.text(cx, yy + sh / 2, label, ha="center", va="center", fontsize=7.5)
            yy -= sh + sgap
        return cont_h

    def draw_arrow(cx, y_from, y_to):
        ax.add_patch(
            FancyArrowPatch(
                (cx, y_from),
                (cx, y_to),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.1,
                color="#333333",
            )
        )

    # ================= Column 1: input + conv stack =================
    c0 = col_x[0]
    y_top_page = 0.0

    # Input at the top.
    ih = 0.72
    draw_box(
        c0,
        y_top_page - ih,
        ih,
        C_IO,
        f"Input   $I_p(t)$\n1 × {model.max_length}",
        bold=True,
    )
    prev_bot = y_top_page - ih

    # Four conv blocks, drawn explicitly (they differ in kernel/channels).
    conv_top, conv_bot = None, None
    for (ch, length), conv in zip(conv_shapes, convs):
        top = prev_bot - arrow_len
        draw_arrow(c0, prev_bot, top)
        conv_sub = [
            (f"Conv1d   k={conv.kernel_size[0]}", C_COMPUTE),
            ("BatchNorm", C_NORM),
            ("ReLU", C_ACT),
            (f"MaxPool   /{pool_k}", C_POOL),
        ]
        cont_h = draw_container(c0, top, conv_sub)
        if conv_top is None:
            conv_top = top
        conv_bot = top - cont_h
        # Output shape of this block, on the left.
        ax.text(
            c0 - (cont_dx + 0.25),
            top - cont_h / 2,
            f"{ch} × {length}",
            ha="right",
            va="center",
            fontsize=7,
            color="#333333",
        )
        prev_bot = top - cont_h

    col1_bot = prev_bot

    # ============ Column 2: flows UPWARD (serpentine second leg) ============
    # Ordered bottom-to-top: Flatten -> FC head -> Linear -> Sigmoid -> output.
    # Build a stack of (kind, payload) and lay it out from col1_bot upward.
    c1 = col_x[1]

    def block_height(kind, payload):
        if kind == "box":
            return 0.6
        # container
        n = len(payload)
        return n * sh + (n - 1) * sgap + 2 * pad

    fc_sub = [
        ("Linear", C_COMPUTE),
        ("BatchNorm", C_NORM),
        ("ReLU", C_ACT),
        ("Dropout", C_DROP),
    ]
    col2 = [
        ("box", (C_DROP, f"Flatten\n{flat_features}")),
        ("container", fc_sub),
        ("container", fc_sub),
        ("box", (C_LINOUT, f"Linear\n{model.fc2.out_features} → 1")),
        ("box", (C_ACT, "Sigmoid")),
    ]

    # Bottom of column 2 aligns with the bottom of the conv stack.
    y = col1_bot
    flatten_bot = y  # remember for the connector
    prev_top = None
    for i, (kind, payload) in enumerate(col2):
        h = block_height(kind, payload)
        y_top = y + h
        if prev_top is not None:
            draw_arrow(c1, prev_top, y)  # arrow points up into this block
        if kind == "box":
            color, label = payload
            draw_box(c1, y, h, color, label)
        else:
            draw_container(c1, y_top, payload)
            ax.text(
                c1 - (cont_dx + 0.25),
                y + h / 2,
                f"{model.fc1.out_features} → {model.fc2.out_features}",
                ha="right",
                va="center",
                fontsize=7,
                color="#333333",
            )
        prev_top = y_top
        y = y_top + arrow_len

    # Output label + arrow at the top of column 2.
    draw_arrow(c1, prev_top, prev_top + arrow_len)
    ax.text(
        c1,
        prev_top + arrow_len + 0.2,
        "Disruption\nProbability",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    col2_top = prev_top + arrow_len + 0.7

    # ---- serpentine connector: bottom of conv stack -> bottom of column 2 ----
    y_route = min(col1_bot, flatten_bot) - arrow_len
    xs = [c0, c0, c1, c1]
    ys = [col1_bot, y_route, y_route, flatten_bot]
    ax.plot(xs[:-1], ys[:-1], color="#333333", lw=1.1, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            (c1, y_route),
            (c1, flatten_bot),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#333333",
        )
    )

    ax.set_xlim(c0 - (cont_dx + 1.9), c1 + (cont_dx + 1.4))
    ax.set_ylim(min(col1_bot, y_route) - 0.5, max(0.4, col2_top))
    span = max(0.4, col2_top) - (min(col1_bot, y_route) - 0.5)
    fig.set_size_inches(7.4, span * 0.5)

    fig_path = Path(fig_path)
    if fig_path.is_dir() or fig_path.suffix == "":
        out_path = fig_path / "model_diagram.png"
    else:
        out_path = fig_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model diagram to {out_path}")
    return out_path


def main() -> None:
    load_best_model_env()
    model_dir = best_model_dir()
    dataset = IpDataset(
        data_file=str(_abs(os.environ["DATA_PATH"])),
        labels_file=str(_abs(os.environ["TRAIN_LABELS_PATH"])),
        labels_path=str(_abs(os.environ["LABELS_PATH"])),
        data_dir=str(_abs(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    model = load_best_model_cnn(dataset)
    if model is None:
        logger.warning(
            "No checkpoint found in best_model/*_best_params.pt; skipping model diagram."
        )
        return

    create_model_diagram(model, model_dir)


if __name__ == "__main__":
    main()
