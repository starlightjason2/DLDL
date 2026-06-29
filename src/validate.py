"""Validation script: runs preprocessed file checks and dataset integrity verification."""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from model.dataset import IpDataset
from util.hptune import load_best_trial_cnn

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(_REPO / p)


os.makedirs(_abs(os.environ["PROG_DIR"]), exist_ok=True)
for _parent in {
    Path(_abs(os.environ["DATA_PATH"])).parent,
    Path(_abs(os.environ["TRAIN_LABELS_PATH"])).parent,
}:
    os.makedirs(_parent, exist_ok=True)

prog_dir = _abs(os.environ["PROG_DIR"])
data_path = _abs(os.environ["DATA_PATH"])
labels_pt_path = _abs(os.environ["TRAIN_LABELS_PATH"])

# Configure logging
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(prog_dir, "validate.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)


def validate_preprocessed_files() -> None:
    """Validate that preprocessed tensor files exist on disk."""
    if not os.path.exists(data_path) or not os.path.exists(labels_pt_path):
        missing = [
            f"Dataset: {data_path}" if not os.path.exists(data_path) else None,
            f"Labels: {labels_pt_path}" if not os.path.exists(labels_pt_path) else None,
        ]
        missing = [m for m in missing if m]
        logger.error(
            "Preprocessed tensor files not found (see DATA_PATH, TRAIN_LABELS_PATH). "
            f"Missing: {', '.join(missing)}"
        )
        raise FileNotFoundError(
            "Preprocessed files not found. Run preprocess_data.py first."
        )
    logger.info("Preprocessed files exist: OK")


def evaluate_best_model(batch_size: int = 256) -> None:
    """Run the best-trial model over the full dataset and log classification
    metrics, plus the shot ids of all false positives and false negatives.
    """
    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=data_path,
        labels_file=labels_pt_path,
        labels_path=_abs(os.environ["LABELS_PATH"]),
        data_dir=_abs(os.environ["DATA_DIR"]),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )
    model = load_best_trial_cnn(dataset)
    if model is None:
        logger.warning(
            "No best-trial checkpoint found (HPTUNE_DIR/trials/best_trial); "
            "skipping model evaluation."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    threshold = model.decision_threshold
    fbeta = float(os.environ.get("FBETA", "1.8"))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    fp_shot_ids: list[int] = []
    fn_shot_ids: list[int] = []

    logger.info(
        "Evaluating best model on {} shots (device={}, threshold={:.4f})...",
        len(dataset),
        device,
        threshold,
    )

    idx = 0
    with torch.no_grad():
        for data, labels in loader:
            outputs = model.forward(data.float().to(device))
            probs = torch.sigmoid(outputs[:, 0]).cpu().numpy()

            preds = (probs > threshold).astype(int)
            cls_true = labels[:, 0].cpu().numpy().astype(int)

            for row in range(len(preds)):
                shot_id = dataset.shot_number(idx)
                if preds[row] == 1 and cls_true[row] == 0:
                    fp_shot_ids.append(shot_id)
                elif preds[row] == 0 and cls_true[row] == 1:
                    fn_shot_ids.append(shot_id)
                idx += 1

            y_true.extend(cls_true.tolist())
            y_pred.extend(preds.tolist())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    logger.info("=" * 60)
    logger.info("Best-model evaluation ({} shots):", len(y_true))
    logger.info("  Accuracy:  {:.6f}", accuracy_score(y_true, y_pred))
    logger.info("  Precision: {:.6f}", precision_score(y_true, y_pred, zero_division=0))
    logger.info("  Recall:    {:.6f}", recall_score(y_true, y_pred, zero_division=0))
    logger.info("  F1:        {:.6f}", f1_score(y_true, y_pred, zero_division=0))
    logger.info(
        "  F{:g}:        {:.6f}",
        fbeta,
        fbeta_score(y_true, y_pred, beta=fbeta, zero_division=0),
    )
    logger.info("  Confusion: TP={} FP={} FN={} TN={}", tp, fp, fn, tn)
    logger.info(
        "  False positives: {} | False negatives: {}",
        len(fp_shot_ids),
        len(fn_shot_ids),
    )
    logger.info("  False-positive shot ids: {}", sorted(fp_shot_ids))
    logger.info("  False-negative shot ids: {}", sorted(fn_shot_ids))
    logger.info("=" * 60)

    fp_file = Path(prog_dir) / "false_positives.txt"
    fn_file = Path(prog_dir) / "false_negatives.txt"
    preds_file = Path(prog_dir) / "predictions.csv"
    fp_file.write_text("\n".join(str(s) for s in sorted(fp_shot_ids)) + "\n")
    fn_file.write_text("\n".join(str(s) for s in sorted(fn_shot_ids)) + "\n")
    preds_file.write_text(
        "predicted_prob, true_prob".join(
            f"{pred},{true}\n" for (pred, true) in zip(y_pred, y_true)
        )
    )
    logger.info(
        "Wrote misclassified shot ids to {} and {}. Wrote predictions data to {}.",
        fp_file,
        fn_file,
        preds_file,
    )


def graph_predictions(preds_file: Path):
    df = pd.read_csv("your_file.csv")

    # 2. Extract columns for X and Y axes
    x_data = df["predicted_prob"]
    y_data = df["true_prob"]

    # 3. Create and customize the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        x_data, y_data, marker="o", linestyle="-", color="b", label="Data Trend"
    )

    # 4. Add titles and axis labels
    plt.xlabel("Probability")
    plt.ylabel("Probability")

    # 5. Add grid and legend, then display
    plt.grid(True)
    plt.legend()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate preprocessed data and dataset integrity"
    )
    parser.add_argument(
        "--skip-model-eval",
        action="store_true",
        help="Skip running the best-trial model over the dataset",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for best-model evaluation (default: 256)",
    )
    args = parser.parse_args()

    logger.info("Running validation...")
    validate_preprocessed_files()

    if not args.skip_model_eval:
        evaluate_best_model(batch_size=args.eval_batch_size)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
