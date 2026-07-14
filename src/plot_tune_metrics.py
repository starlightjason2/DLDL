"""Plot recall, precision, and F2 vs trial number from HP tune ``trials.csv`` files."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from model.trial_status import TrialStatus
from util.best_model import repo_root
from util.objective import fbeta_from_pr

_TUNE_SOURCES = (
    ("ARCH_TUNE_DIR", "Architecture HP tune"),
    ("HP_TUNE_DIR", "HP tune"),
)


def _trial_num(trial_id: str) -> int:
    match = re.fullmatch(r"trial_(\d+)", str(trial_id).strip())
    if not match:
        raise ValueError(f"unexpected trial_id: {trial_id!r}")
    return int(match.group(1))


def _plot(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == TrialStatus.COMPLETED].copy()
    if df.empty:
        print(f"Skipping {csv_path}: no completed trials")
        return

    df["trial_num"] = df["trial_id"].map(_trial_num)
    df["f2"] = [
        fbeta_from_pr(float(recall), float(precision), beta=2.0)
        for recall, precision in zip(df["recall"], df["precision"])
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["trial_num"], df["recall"], label="Recall", alpha=0.85, s=40)
    ax.scatter(df["trial_num"], df["precision"], label="Precision", alpha=0.85, s=40)
    ax.scatter(df["trial_num"], df["f2"], label="F2", alpha=0.85, s=40)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    root = repo_root()
    load_dotenv(root / ".env", encoding="utf-8")

    for env_key, label in _TUNE_SOURCES:
        tune_dir = Path(os.environ[env_key])
        if not tune_dir.is_absolute():
            tune_dir = root / tune_dir
        csv_path = tune_dir / "trials" / "trials.csv"
        if not csv_path.exists():
            print(f"Skipping {env_key}: missing {csv_path}")
            continue
        _plot(
            csv_path,
            tune_dir / "trials" / "tune_metrics.png",
        )


if __name__ == "__main__":
    main()
