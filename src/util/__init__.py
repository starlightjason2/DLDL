"""Utility module: data loading and preprocessing."""

import sys
from loguru import logger
from .data_loading import (
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad_norm,
    env_float,
    env_int,
    env_tuple,
)
from .processing import (
    get_use_cores,
    create_binary_labels,
    convert_tensors_to_float,
)

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="INFO",
)


__all__ = [
    "env_float",
    "env_int",
    "env_tuple",
    "get_length",
    "get_scaled_t_disrupt",
    "get_means",
    "load_and_pad_norm",
    "get_use_cores",
    "create_binary_labels",
    "convert_tensors_to_float",
]
