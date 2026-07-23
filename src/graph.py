"""Browse preprocessed shots with a slider and index text box."""

import argparse
import math
import os
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider, TextBox

from model.dataset import IpDataset
from util.data_loading import _read_signal_file
from util.disruption_predict import (
    predict_disruption_time,
    apply_filter,
    get_oriented_current,
    clean_zeros,
)
from util.best_model import load_best_model_cnn, load_best_model_env


def _build_dataset() -> IpDataset:
    repo = Path(__file__).resolve().parents[1]
    load_best_model_env()

    def abs_path(p: str) -> str:
        return p if os.path.isabs(p) else str(repo / p)

    return IpDataset(
        data_file=abs_path(os.environ["DATA_PATH"]),
        labels_file=abs_path(os.environ["TRAIN_LABELS_PATH"]),
        labels_path=abs_path(os.environ["LABELS_PATH"]),
        data_dir=abs_path(os.environ["DATA_DIR"]),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )


def _make_draw(dataset: IpDataset, model, num_rows: int):
    """Build the per-shot draw function bound to a dataset/model.

    The returned ``draw(ax1, ax2, i)`` renders one shot into the current-signal
    axis ``ax1`` and the heuristic axis ``ax2``. It returns the shot view so
    callers can title panels as they see fit.
    """

    def draw(ax1, ax2, i: int):
        shot = dataset.load_shot_view(i)

        idx = max(0, min(int(i), num_rows - 1))
        signal = dataset.data[idx].float().reshape(1, -1)
        cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()

        # Plot the raw shot file directly: column 0 is time (s), column 1 is
        # current (SI), so the axes carry physical units without de-normalizing.
        raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
        raw_current = _read_signal_file(raw_path, col=1)
        raw_time = _read_signal_file(raw_path, col=0)
        current, time = clean_zeros(raw_current, raw_time)
        predicted_time_start, pred_time, predicted_time_end = predict_disruption_time(
            raw_current, raw_time
        )

        # t_disrupt is stored normalized (disruption_index / max_length); map it
        # back onto the SI time axis via the raw time samples.
        max_length = dataset.data.shape[1]
        t_disrupt_si = (
            float(raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)])
            if shot.disruptive
            else None
        )

        ax1.clear()
        ax1.set_title(f"CNN disruption probability: {100*cnn_prob:.2f}%", fontsize=10)
        ax1.plot(time, current, label="Current $I(t)$")
        flipped_current = get_oriented_current(current)
        if not np.array_equal(current, flipped_current):
            ax1.plot(
                time,
                flipped_current,
                color="C0",
                label="Flipped $I_\\mathrm{raw}(t)$",
                linestyle=":",
            )

        filtered, smoothed = apply_filter(current)
        ax1.plot(
            time,
            smoothed,
            color="C1",
            label="Smoothed $I_\\mathrm{smooth}(t)$",
            linestyle="--",
        )
        if shot.disruptive:
            ax1.axvline(
                t_disrupt_si,
                color="black",
                ls="--",
                linewidth=1,
                label=f"Real disruption time: $t_0={t_disrupt_si:.3f}$s",
            )
            ax2.axvline(
                t_disrupt_si,
                color="black",
                ls="--",
                linewidth=1,
                label=f"Real disruption time: $t_0={t_disrupt_si:.3f}$s",
            )

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Current (MA)")
        ax1.legend(loc="lower left")
        ax1.grid(True)

        ax2.clear()
        ax2.set_ylabel("Disruption Heurisitic $f(t)$")
        ax2.plot(time, filtered, label="Filter")
        diff = (
            (t_disrupt_si - predicted_time_start) if t_disrupt_si is not None else None
        )
        heuristic_label = (
            f"Heuristic disruption time:\n$t={pred_time:.3f}$s, "
            f"{f'{diff:.4f} s diff' if diff else ''}"
        )
        ax2.axvline(pred_time, color="C3", ls="--", linewidth=1, label=heuristic_label)
        ax1.axvline(pred_time, color="C3", ls="--", linewidth=1, label=heuristic_label)
        ax2.axvspan(
            xmin=predicted_time_start,
            xmax=predicted_time_end,
            alpha=0.2,
            label="Heuristic Disruption Interval",
            color="C3",
        )
        ax1.axvspan(
            xmin=predicted_time_start,
            xmax=predicted_time_end,
            alpha=0.2,
            label="Heuristic Disruption Interval",
            color="C3",
        )
        ax2.legend(loc="lower left")
        ax2.grid()

        # Zoom to a window centered on the disruption (true time if known,
        # else the heuristic prediction), clamped to the available time range.
        window = 0.1
        center = t_disrupt_si if t_disrupt_si is not None else pred_time
        lo = max(time[0], center - window / 2)
        hi = min(time[-1], center + window / 2)
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(-0.1, 1.2 * current[time == predicted_time_start][0])
        ax2.set_xlim(lo, hi)

        return shot

    return draw


def run_interactive(dataset: IpDataset, model, num_rows: int) -> None:
    """Slider/text-box browser over all shots (one shot at a time)."""
    matplotlib.use("QtAgg")
    draw = _make_draw(dataset, model, num_rows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.subplots_adjust(bottom=0.2)

    with torch.no_grad():

        def redraw(i: int) -> None:
            shot = draw(ax1, ax2, i)
            fig.suptitle(shot.title)
            fig.canvas.draw_idle()

        redraw(0)

        index_slider = Slider(
            fig.add_axes([0.12, 0.05, 0.55, 0.03]),
            "index",
            0,
            num_rows - 1,
            valinit=0,
            valstep=1,
        )
        index_slider.valtext.set_visible(False)
        box = TextBox(fig.add_axes([0.72, 0.05, 0.12, 0.03]), "", initial="0")

        index_slider.on_changed(lambda v: (redraw(v), box.set_val(str(int(v)))))
        box.on_submit(
            lambda t: index_slider.set_val(int(t)) if t.strip().isdigit() else None
        )

        plt.show()


def save_grid(
    dataset: IpDataset,
    model,
    num_rows: int,
    indices: list[int],
) -> None:
    """Render the given shot indices as a grid of (current, heuristic) panels.

    Each shot occupies one grid column with two stacked rows: the current-signal
    axis on top and the heuristic axis below, reusing the shared draw function.
    Saves to ``shot_grid.png``.
    """
    matplotlib.use("Agg")
    draw = _make_draw(dataset, model, num_rows)
    out_path = Path("shot_grid.png")

    n = len(indices)
    ncols = min(n, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    # Two physical rows (current + heuristic) per shot row.
    fig, axes = plt.subplots(
        nrows * 2,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
    )

    with torch.no_grad():
        for cell, i in enumerate(indices):
            row, col = divmod(cell, ncols)
            ax1 = axes[row * 2][col]
            ax2 = axes[row * 2 + 1][col]
            shot = draw(ax1, ax2, i)
            # draw() sets ax1's title to the CNN probability; prepend the shot id.
            ax1.set_title(f"{shot.title}\n{ax1.get_title()}", fontsize=9)

    # Blank any unused panels in the final grid row.
    for cell in range(n, nrows * ncols):
        row, col = divmod(cell, ncols)
        axes[row * 2][col].axis("off")
        axes[row * 2 + 1][col].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse preprocessed shots, or save a grid of specific shots."
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="IDX",
        help="Shot indices to render as a grid of subplots instead of browsing.",
    )
    args = parser.parse_args()

    dataset = _build_dataset()
    num_rows = len(dataset)

    model = load_best_model_cnn(dataset)
    if model is None:
        return

    if args.indices:
        save_grid(dataset, model, num_rows, args.indices)
    else:
        run_interactive(dataset, model, num_rows)


if __name__ == "__main__":
    main()
