"""Data preprocessing script for DLDL."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from config.schema import DatasetEnv

_REPO = Path(__file__).resolve().parents[1]
if (_env := _REPO / ".env").is_file():
    load_dotenv(_env)


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

# Configure logging: stderr + file
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(prog_dir, "preprocess.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
from model.dataset import IpDataset

if __name__ == "__main__":
    n = DatasetEnv.from_os().normalization_type
    logger.info("Cleaning up cached preprocessed files...")
    for path in (data_path, labels_pt_path):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted cached file: {path}")
    IpDataset(normalization_type=n).check_dataset(scale_labels=True)
