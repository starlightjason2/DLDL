"""Plot the ROC curve for the best_model network on the dev holdout set."""

from __future__ import annotations

import os

import torch

# Prevent macOS OpenMP thread deadlock on CPU inference
if not torch.cuda.is_available():
    torch.set_num_threads(1)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.dataset import IpDataset
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(str(best_model_dir().parent), path)


@torch.no_grad()
def _collect_scores(
    model: torch.nn.Module,
    dev: Subset,
    *,
    batch_size: int,
    device: str,
) -> tuple[list[int], list[float]]:
    """Run the model over the dev set, returning true labels and disruption scores."""
    loader = DataLoader(dev, batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_score: list[float] = []
    for signals, labels in tqdm(loader, desc="Scoring", unit="batch"):
        probs = torch.sigmoid(model(signals.float().to(device))[:, 0]).cpu().numpy()
        y_score.extend(probs.tolist())
        y_true.extend(labels[:, 0].cpu().numpy().astype(int).tolist())
    return y_true, y_score


def plot_roc_curve(batch_size: int = 256) -> None:
    load_best_model_env()
    model_dir = best_model_dir()

    dataset = IpDataset(
        data_file=_abs(os.environ["DATA_PATH"]),
        labels_file=_abs(os.environ["TRAIN_LABELS_PATH"]),
        labels_path=_abs(os.environ["LABELS_PATH"]),
        data_dir=_abs(os.environ["DATA_DIR"]),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )
    _, dev, _ = dataset.split()

    model = load_best_model_cnn(dataset)
    if model is None:
        raise FileNotFoundError(
            "No checkpoint found in best_model/*_best_params.pt. Train a model first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        batch_size = min(batch_size, 8)  # 60k-length sequences are huge on CPU
    model = model.to(device)

    y_true, y_score = _collect_scores(model, dev, batch_size=batch_size, device=device)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC AUC (n={len(y_true)} dev holdout shots): {auc:.6f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f"IpCNN (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — disruption classifier (dev holdout)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = model_dir / "roc_curve.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_roc_curve()
