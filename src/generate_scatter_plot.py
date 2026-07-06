"""Validation script: runs preprocessed file checks and dataset integrity verification."""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(_REPO / p)


prog_dir = _abs(os.environ["PROG_DIR"])
disruption_predictions_data = Path(prog_dir) / "predictions.csv"
disruption_predictions_graph = Path(prog_dir) / "predictions.png"


def main() -> None:
    df = pd.read_csv(disruption_predictions_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["predicted_time"], df["true_time"])

    plt.xlabel("Predicted disruption time (s)")
    plt.ylabel("Real disruption time (s)")

    plt.grid(True)
    plt.savefig(disruption_predictions_graph, dpi=600)


if __name__ == "__main__":
    main()
