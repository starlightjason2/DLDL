"""Evaluate the best model on the dev holdout set."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
from torch.utils.data import DataLoader, Subset

from model.dataset import IpDataset
from util.disruption_predict import predict_disruption_time
from util.training import load_best_epoch_cnn

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(_REPO / ".env", encoding="utf-8")


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def _configure_logging(prog_dir: Path) -> None:
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stderr, format=fmt, colorize=True, level="INFO")
    logger.add(
        prog_dir / "validate.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )


def _require_preprocessed(data_path: Path, labels_path: Path) -> None:
    missing = [p for p in (data_path, labels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed files not found. Run preprocess_data.py first. "
            f"Missing: {', '.join(map(str, missing))}"
        )
    logger.info("Preprocessed files exist: OK")


@dataclass
class EvalResult:
    y_true: list[int]
    y_pred: list[int]
    fp_shot_ids: list[int]
    fn_shot_ids: list[int]
    disruption_times: list[tuple[float, float]]


def _evaluate_dev(
    dataset: IpDataset,
    model: torch.nn.Module,
    dev: Subset,
    *,
    batch_size: int,
    device: str,
) -> EvalResult:
    dev_indices = list(dev.indices)
    loader = DataLoader(dev, batch_size=batch_size, shuffle=False)
    threshold = model.decision_threshold
    y_true: list[int] = []
    y_pred: list[int] = []
    fp_shot_ids: list[int] = []
    fn_shot_ids: list[int] = []
    disruption_times: list[tuple[float, float]] = []
    offset = 0

    with torch.no_grad():
        for signals, labels in loader:
            probs = torch.sigmoid(model(signals.float().to(device))[:, 0]).cpu().numpy()
            preds = (probs > threshold).astype(int)
            actuals = labels[:, 0].cpu().numpy().astype(int)

            for i, (pred, actual) in enumerate(zip(preds, actuals)):
                idx = dev_indices[offset + i]
                if pred != actual:
                    (fp_shot_ids if pred else fn_shot_ids).append(idx)
                if actual:
                    shot = dataset.load_shot_view(idx)
                    pred_time = predict_disruption_time(shot.current, shot.time)
                    disruption_times.append((pred_time, shot.t_disrupt))

            offset += len(preds)
            y_true.extend(actuals.tolist())
            y_pred.extend(preds.tolist())

    return EvalResult(y_true, y_pred, fp_shot_ids, fn_shot_ids, disruption_times)


def _log_metrics(result: EvalResult, *, fbeta: float) -> None:
    tn, fp, fn, tp = confusion_matrix(
        result.y_true, result.y_pred, labels=[0, 1]
    ).ravel()
    logger.info("=" * 60)
    logger.info("Best-model evaluation ({} dev holdout shots):", len(result.y_true))
    logger.info("  Accuracy:  {:.6f}", accuracy_score(result.y_true, result.y_pred))
    logger.info(
        "  Precision: {:.6f}",
        precision_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  Recall:    {:.6f}",
        recall_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  F1:        {:.6f}",
        f1_score(result.y_true, result.y_pred, zero_division=0),
    )
    logger.info(
        "  F{:g}:        {:.6f}",
        fbeta,
        fbeta_score(result.y_true, result.y_pred, beta=fbeta, zero_division=0),
    )
    logger.info("  Confusion: TP={} FP={} FN={} TN={}", tp, fp, fn, tn)
    logger.info(
        "  False positives: {} | False negatives: {}",
        len(result.fp_shot_ids),
        len(result.fn_shot_ids),
    )
    logger.info("  False-positive shot ids: {}", sorted(result.fp_shot_ids))
    logger.info("  False-negative shot ids: {}", sorted(result.fn_shot_ids))
    logger.info("=" * 60)


def _write_artifacts(prog_dir: Path, result: EvalResult) -> None:
    (prog_dir / "false_positives.txt").write_text(
        "\n".join(map(str, sorted(result.fp_shot_ids))) + "\n"
    )
    (prog_dir / "false_negatives.txt").write_text(
        "\n".join(map(str, sorted(result.fn_shot_ids))) + "\n"
    )
    rows = ["predicted_time,true_time"] + [
        f"{pred},{true}" for pred, true in result.disruption_times
    ]
    predictions_csv = prog_dir / "predictions.csv"
    predictions_csv.write_text("\n".join(rows) + "\n")
    logger.info(
        "Wrote misclassified shot ids and {} disruption-time pairs to {}",
        len(result.disruption_times),
        prog_dir,
    )


def evaluate_best_model(prog_dir: Path, batch_size: int = 256) -> None:
    """Run the best checkpoint on the dev holdout used during training."""
    dataset = IpDataset(
        data_file=str(_abs(os.environ["DATA_PATH"])),
        labels_file=str(_abs(os.environ["TRAIN_LABELS_PATH"])),
        labels_path=str(_abs(os.environ["LABELS_PATH"])),
        data_dir=str(_abs(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    _, dev, _ = dataset.split()

    model = load_best_epoch_cnn(dataset)
    if model is None:
        logger.warning(
            "No best checkpoint found (HPTUNE_DIR/trials/best_trial or "
            "PROG_DIR/best_epoch); skipping model evaluation."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    fbeta = float(os.environ.get("FBETA", "1.8"))

    logger.info(
        "Evaluating best model on {} dev holdout shots (device={}, threshold={:.4f})",
        len(dev),
        device,
        model.decision_threshold,
    )

    result = _evaluate_dev(dataset, model, dev, batch_size=batch_size, device=device)
    _log_metrics(result, fbeta=fbeta)
    _write_artifacts(prog_dir, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate data and evaluate the best model"
    )
    parser.add_argument(
        "--skip-model-eval",
        action="store_true",
        help="Skip running the best model on the dev holdout set",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for model evaluation (default: 256)",
    )
    args = parser.parse_args()

    prog_dir = _abs(os.environ["PROG_DIR"])
    data_path = _abs(os.environ["DATA_PATH"])
    labels_path = _abs(os.environ["TRAIN_LABELS_PATH"])
    for path in (prog_dir, data_path.parent, labels_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    _configure_logging(prog_dir)
    logger.info("Running validation...")
    _require_preprocessed(data_path, labels_path)

    if not args.skip_model_eval:
        evaluate_best_model(prog_dir, batch_size=args.eval_batch_size)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
