"""
Utility module for DLDL project.

This package provides utility functions for:
- File I/O operations
- Data loading and preprocessing
- Dataset manipulation (splitting)
- Distributed training setup/cleanup
- Data preprocessing pipeline (Preprocessor class)

Exports:
    - Preprocessor: Main class for preprocessing raw signal data
    - Various utility functions for data loading and processing
    - Distributed training utilities
"""

from .utils import (
    check_file,
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad,
    load_and_pad_norm,
    load_and_pad_scale,
    split,
    setup,
    setup_file,
    cleanup,
)
from .preprocessor import Preprocessor

__all__ = [
    "check_file",
    "get_length",
    "get_scaled_t_disrupt",
    "get_means",
    "load_and_pad",
    "load_and_pad_norm",
    "load_and_pad_scale",
    "split",
    "setup",
    "setup_file",
    "cleanup",
    "Preprocessor",
]
