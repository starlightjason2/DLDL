"""Browse preprocessed shots with a slider and index text box."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np
import torch
from matplotlib.widgets import Slider, TextBox

from model.dataset import IpDataset
from util.disruption_predict import predict_disruption_time, apply_filter, apply_smoothing
from util.hptune import load_best_trial_cnn

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    repo = Path(__file__).resolve().parents[1]

    def abs_path(p: str) -> str:
        return p if os.path.isabs(p) else str(repo / p)

    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=abs_path(os.environ["DATA_PATH"]),
        labels_file=abs_path(os.environ["TRAIN_LABELS_PATH"]),
        labels_path=abs_path(os.environ["LABELS_PATH"]),
        data_dir=abs_path(os.environ["DATA_DIR"]),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )
    num_rows = len(dataset)

    model = load_best_trial_cnn(dataset)
    if model is None:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(bottom=0.2)

    with torch.no_grad():

        def draw(i: int) -> None:
            shot = dataset.load_shot_view(i)
            fig.suptitle(shot.title)
            
            idx = max(0, min(int(i), num_rows - 1))
            signal = dataset.data[idx].float().reshape(1, -1)
            cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()
            predicted_time = predict_disruption_time(shot)

            ax1.clear()
            ax1.set_title(
                f"CNN disruption probability: {100*cnn_prob:.2f}%", fontsize=10
            )
            ax1.plot(shot.time, shot.current, label="Current $I(t)$")  
            ax1.plot(shot.time, apply_smoothing(shot.current), label="Smoothed $I(t)$", linestyle="--")  
            if shot.disruptive:
                ax1.axvline(
                    shot.t_disrupt,
                    color="r",
                    ls="--",
                    label=f"Real disruption time: $t_0={shot.t_disrupt:.5f}$s",
                )
        
            ax1.set_xlabel("Normalized time")
            ax1.set_ylabel("Normalized current")
            ax1.legend()
            ax1.grid(True)

            ax2.clear()
            ax2.plot(shot.time, np.gradient(apply_smoothing(shot.current), shot.time), label="$dI/dt$")
            ax2.legend()
            ax2.grid(True)

            ax3.clear()
            ax3.plot(shot.time, apply_filter(shot), label="Filter")
            diff_microsec = abs(shot.t_disrupt - predicted_time) * 1e5 if shot.t_disrupt is not None else None
            ax3.axvline(
                predicted_time,
                color="r",
                ls="--",
                label=f"Heuristic disruption time:\n$t={predicted_time:.5f}$s, {f"{diff_microsec:.1f} µs diff" if diff_microsec else ""}",
            )    
            
            ax3.legend()
            ax3.grid()

            fig.canvas.draw_idle()


        draw(start)

        index_slider = Slider(
            fig.add_axes([0.12, 0.05, 0.55, 0.03]),
            "Index",
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
