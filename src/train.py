"""Training entry point for DLDL disruption prediction model."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from config.settings import Settings
from schemas.trial_schema import cleanup_epoch_checkpoints, submit_next_serial_controller

_REPO = Path(__file__).resolve().parents[1]  # project root (src/..)
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(_REPO / p)


os.makedirs(_abs(os.environ["PROG_DIR"]), exist_ok=True)
for _parent in {
    Path(_abs(os.environ["DATA_PATH"])).parent,
    Path(_abs(os.environ["TRAIN_LABELS_PATH"])).parent,
}:
    os.makedirs(_parent, exist_ok=True)

s = Settings.load()
prog_dir = _abs(os.environ["PROG_DIR"])
job_id = os.environ["JOB_ID"]
data_path = _abs(os.environ["DATA_PATH"])
labels_pt_path = _abs(os.environ["TRAIN_LABELS_PATH"])

# Configure logging: stderr + optional file (HPTune sets TRAIN_LOGURU_FILE=0 when tee captures stderr)
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, colorize=True, level="INFO")
if os.environ.get("TRAIN_LOGURU_FILE", "1").lower() not in ("0", "false", "no"):
    logger.add(
        os.path.join(prog_dir, f"{job_id}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )

from model.cnn import IpCNN

if __name__ == "__main__":
    a, t = s.architecture_config, s.training_config
    model = IpCNN(
        data_path=data_path,
        labels_path=labels_pt_path,
        prog_dir=prog_dir,
        conv1=(a.conv1_filters, a.conv1_kernel, a.conv1_padding),
        conv2=(a.conv2_filters, a.conv2_kernel, a.conv2_padding),
        conv3=(a.conv3_filters, a.conv3_kernel, a.conv3_padding),
        pool_size=a.pool_size,
        fc1_size=a.fc1_size,
        fc2_size=a.fc2_size,
        dropout_rate=t.dropout_rate,
        classification=False,
        normalization_type=t.normalization_type,
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model.train_model(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        job_id=job_id,
    )
    if rank == 0:
        cleanup_epoch_checkpoints(prog_dir, job_id)
        submit_next_serial_controller()
