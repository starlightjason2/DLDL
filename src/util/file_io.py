"""File I/O utilities for DLDL project."""

import os


def check_file(file_path: str, verbose: bool = False) -> bool:
    """Check if file exists. If verbose, print file size or non-existence message."""
    if os.path.exists(file_path):
        if verbose:
            file_size: int = os.path.getsize(file_path)
            print(f"File {file_path} exists. Size: {file_size} bytes.")
        return True
    else:
        if verbose:
            print(f"File {file_path} does not exist.")
        return False
