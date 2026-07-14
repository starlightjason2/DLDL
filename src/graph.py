"""Browse preprocessed shots with a slider and index text box."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
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


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    repo = Path(__file__).resolve().parents[1]
    load_best_model_env()

    def abs_path(p: str) -> str:
        return p if os.path.isabs(p) else str(repo / p)

    dataset = IpDataset(
        data_file=abs_path(os.environ["DATA_PATH"]),
        labels_file=abs_path(os.environ["TRAIN_LABELS_PATH"]),
        labels_path=abs_path(os.environ["LABELS_PATH"]),
        data_dir=abs_path(os.environ["DATA_DIR"]),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )
    num_rows = len(dataset)

    model = load_best_model_cnn(dataset)
    if model is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.subplots_adjust(bottom=0.2)

    with torch.no_grad():

        def draw(i: int) -> None:
            shot = dataset.load_shot_view(i)
            fig.suptitle(shot.title)

            idx = max(0, min(int(i), num_rows - 1))
            signal = dataset.data[idx].float().reshape(1, -1)
            cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()

            # Plot the raw shot file directly: column 0 is time (s), column 1 is
            # current (SI), so the axes carry physical units without de-normalizing.
            raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
            raw_current = _read_signal_file(raw_path, col=1)
            raw_time = _read_signal_file(raw_path, col=0)
            current, time = clean_zeros(raw_current, raw_time)
            predicted_time = predict_disruption_time(raw_current, raw_time)

            # t_disrupt is stored normalized (disruption_index / max_length); map it
            # back onto the SI time axis via the raw time samples.
            max_length = dataset.data.shape[1]
            t_disrupt_si = (
                float(
                    raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)]
                )
                if shot.disruptive
                else None
            )

            ax1.clear()
            ax1.set_title(
                f"CNN disruption probability: {100*cnn_prob:.2f}%", fontsize=10
            )
            ax1.plot(time, current, label="Current $I(t)$", color="tab:blue")
            flipped = get_oriented_current(current)
            if not np.array_equal(current, flipped):
                ax1.plot(
                    time,
                    flipped,
                    label="Flipped $I(t)$",
                    linestyle=":",
                    color="tab:blue",
                )

            filtered, smoothed = apply_filter(current)
            ax1.plot(
                time,
                smoothed,
                label="Smoothed $I(t)$",
                linestyle="--",
                color="tab:orange",
            )
            if shot.disruptive:
                ax1.axvline(
                    t_disrupt_si,
                    color="r",
                    ls="--",
                    linewidth=1,
                    label=f"Real disruption time: $t_0={t_disrupt_si:.3f}$s",
                )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Current (MA)")
            ax1.legend()
            ax1.grid(True)

            ax2.clear()
            ax2.plot(time, filtered, label="Filter")
            diff = (t_disrupt_si - predicted_time) if t_disrupt_si is not None else None
            ax2.axvline(
                predicted_time,
                color="r",
                ls="--",
                linewidth=1,
                label=f"Heuristic disruption time:\n$t={predicted_time:.3f}$s, {f"{diff:.4f} s diff" if diff else ""}",
            )

            ax2.legend()
            ax2.grid()

            fig.canvas.draw_idle()

        draw(start)

        index_slider = Slider(
            fig.add_axes([0.12, 0.05, 0.55, 0.03]),
            "index",
            0,
            num_rows - 1,
            valinit=start,
            valstep=1,
        )
        index_slider.valtext.set_visible(False)
        box = TextBox(fig.add_axes([0.72, 0.05, 0.12, 0.03]), "", initial=str(start))

        index_slider.on_changed(lambda v: (draw(v), box.set_val(str(int(v)))))
        box.on_submit(
            lambda t: index_slider.set_val(int(t)) if t.strip().isdigit() else None
        )

        plt.show()


if __name__ == "__main__":
    main()
