"""Smoke test: threshold tuning on a real checkpoint dev-set forward pass."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")
sys.path.insert(0, str(_REPO / "src"))

from model.dataset import IpDataset  # noqa: E402
from util.objective import best_threshold, min_precision, score  # noqa: E402
from util.training import build_cnn_from_env, load_checkpoint_into_model  # noqa: E402


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def main() -> None:
    prog_dir = _abs(os.environ["PROG_DIR"])
    checkpoint = prog_dir / "prec90rec_best_params.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint for smoke test: {checkpoint}")

    os.environ.setdefault("JOB_ID", "prec90rec")
    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=str(_abs(os.environ["DATA_PATH"])),
        labels_file=str(_abs(os.environ["TRAIN_LABELS_PATH"])),
        labels_path=str(_abs(os.environ["LABELS_PATH"])),
        data_dir=str(_abs(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )
    _, dev, _ = dataset.split()
    loader = DataLoader(dev, batch_size=256, shuffle=False)

    model = build_cnn_from_env(dataset, str(prog_dir))
    load_checkpoint_into_model(model, checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    labels: list[float] = []
    probs: list[float] = []
    with torch.no_grad():
        for data, targets in loader:
            output = model(data.float().to(device))
            labels.extend(targets[:, 0].cpu().numpy().tolist())
            probs.extend(torch.sigmoid(output[:, 0]).cpu().numpy().tolist())

    threshold, precision, recall = best_threshold(labels, probs)
    objective = score(recall, precision)

    logger.info("Smoke checkpoint: {}", checkpoint.name)
    logger.info("Dev-set size: {}", len(labels))
    logger.info("MIN_PRECISION: {:.4f}", min_precision())
    logger.info("Tuned threshold: {:.6f}", threshold)
    logger.info("Precision: {:.6f}", precision)
    logger.info("Recall:    {:.6f}", recall)
    logger.info("Objective score: {:.6f}", objective)

    if not (0.01 <= threshold <= 0.99):
        raise SystemExit(f"Threshold out of range: {threshold}")
    if recall == 1.0 and precision < 0.5:
        raise SystemExit("Degenerate predict-all threshold selected")
    if precision < min_precision():
        raise SystemExit(
            f"No feasible operating point: precision {precision:.4f} < {min_precision():.4f}"
        )
    if objective <= 0:
        raise SystemExit(f"Objective infeasible: {objective}")

    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
