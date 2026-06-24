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

from model.cnn import IpCNN
from model.dataset import IpDataset
from util.disruption_predict import predict_disruption_time

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


def load_best_cnn(dataset, repo, abs_path):
    """Build an IpCNN with the best trial's weights, or return None if absent."""
    ckpts = sorted(
        (Path(abs_path(os.environ["HPTUNE_DIR"])) / "trials" / "best_trial").glob(
            "*_best_params.pt"
        )
    )
    if not ckpts:
        return None

    model = IpCNN(
        dataset,
        prog_dir=str(repo),
        conv1=(
            int(os.environ["CONV1_FILTERS"]),
            int(os.environ["CONV1_KERNEL"]),
            int(os.environ["CONV1_PADDING"]),
        ),
        conv2=(
            int(os.environ["CONV2_FILTERS"]),
            int(os.environ["CONV2_KERNEL"]),
            int(os.environ["CONV2_PADDING"]),
        ),
        conv3=(
            int(os.environ["CONV3_FILTERS"]),
            int(os.environ["CONV3_KERNEL"]),
            int(os.environ["CONV3_PADDING"]),
        ),
        conv4=(
            int(os.environ["CONV4_FILTERS"]),
            int(os.environ["CONV4_KERNEL"]),
            int(os.environ["CONV4_PADDING"]),
        ),
        pool_size=int(os.environ["POOL_SIZE"]),
        fc1_size=int(os.environ["FC1_SIZE"]),
        fc2_size=int(os.environ["FC2_SIZE"]),
        dropout_rate=float(os.environ["DROPOUT_RATE"]),
        cls_pos_weight=float(os.environ["CLS_POS_WEIGHT"]),
        decision_threshold=float(os.environ["DECISION_THRESHOLD"]),
    )
    model.load_state_dict(torch.load(ckpts[0], map_location="cpu"))
    model.eval()
    return model


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

    model = load_best_cnn(dataset, repo, abs_path)
    if model is None: return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.subplots_adjust(bottom=0.2)
    
    with torch.no_grad():
        
        def draw(i: int) -> None:
            shot = dataset.load_shot_view(i)
        
            idx = max(0, min(int(i), num_rows - 1))
            signal = dataset.data[idx].float().reshape(1, -1)
            cls_logit, cnn_time = model.forward(signal)[0]

            ax1.clear()
            
            ax1.plot(shot.time, shot.current, label="Current")
            if shot.disruptive:
                ax1.axvline(shot.t_disrupt, color="r", ls="--", label=f"Real disruption time: {shot.t_disrupt:.5f}s")
            fig.suptitle(shot.title)
            

            ax1.set_xlabel("Normalized time")
            ax1.set_ylabel("Normalized current")

            window_size = len(shot.time) // 100
            weights = np.ones(window_size) / window_size
            smoothed_current = np.convolve(shot.current, weights, mode="same")
            ax1.plot(shot.time, smoothed_current, label="Smoothed Current", ls="--")
            ax1.grid(True)

            ax2.clear()        
            
            diff = np.abs(shot.current - smoothed_current)
            ax2.plot(shot.time, diff, linewidth=2)                
            predicted_time = predict_disruption_time(shot.current)
           
            cnn_prob = torch.sigmoid(cls_logit).item()
            cnn_disruptive = cnn_prob > model.decision_threshold

            ax1.set_title(f"CNN disruption probability: {100*cnn_prob:.2f}%", fontsize=10)

            if cnn_disruptive:
                ax2.axvline(predicted_time, color="r", ls="--", label=f"Heuristic disruption time: {predicted_time:.5f}s, {abs(shot.t_disrupt - predicted_time) * 1e5:.2f} microsecond diff")
                ax2.axvline(
                    float(cnn_time),
                    color="g",
                    ls=":",
                    label=f"CNN disruption time",
                )

            ax1.legend()
            ax2.legend()
            ax2.grid(True)        
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
