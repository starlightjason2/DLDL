"""Data preprocessing script for DLDL."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from model.dataset import IpDataset

_REPO = Path(__file__).resolve().parents[1]
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")

os.makedirs(os.environ["PROG_DIR"], exist_ok=True)
for _parent in {
    Path(os.environ["DATA_PATH"]).parent,
    Path(os.environ["TRAIN_LABELS_PATH"]).parent,
}:
    os.makedirs(_parent, exist_ok=True)

prog_dir = os.environ["PROG_DIR"]
data_path = os.environ["DATA_PATH"]
labels_pt_path = os.environ["TRAIN_LABELS_PATH"]

# Configure logging: stderr + file
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
logger.add(
    os.path.join(prog_dir, "preprocess.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)


if __name__ == "__main__":
    logger.info("Cleaning up cached preprocessed files...")
    for path in (data_path, labels_pt_path):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted cached file: {path}")
    IpDataset(
        
        data_file=data_path,
        labels_file=labels_pt_path,
        labels_path=os.environ["LABELS_PATH"],
        data_dir=os.environ["DATA_DIR"],
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    ).check_dataset(scale_labels=True)
