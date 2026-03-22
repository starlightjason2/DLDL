from pathlib import Path

import numpy as np
import pytest

from helpers import load_module_from_path


_MODULE = load_module_from_path(
    "test_util_data_loading",
    Path(__file__).resolve().parents[1] / "src" / "util" / "data_loading.py",
)
get_length = _MODULE.get_length
get_means = _MODULE.get_means
get_scaled_t_disrupt = _MODULE.get_scaled_t_disrupt
load_and_pad = _MODULE.load_and_pad
load_and_pad_norm = _MODULE.load_and_pad_norm
load_and_pad_scale = _MODULE.load_and_pad_scale


def write_signal_file(path: Path, rows: list[tuple[float, float]]) -> None:
    path.write_text("\n".join(f"{t} {v}" for t, v in rows), encoding="utf-8")


def test_get_length_and_means(tmp_path: Path) -> None:
    """Read signal-file length and first/second moments from a tiny synthetic file."""
    write_signal_file(tmp_path / "1001.txt", [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)])

    assert get_length("1001.txt", str(tmp_path)) == 3
    assert get_means("1001.txt", str(tmp_path)) == pytest.approx([3.0, 35.0 / 3.0])


def test_get_scaled_t_disrupt_requires_positive_max_length(tmp_path: Path) -> None:
    """Reject scaled disruption requests when ``max_length`` is not positive."""
    write_signal_file(tmp_path / "1002.txt", [(0.0, 1.0), (1.0, 2.0)])

    with pytest.raises(ValueError, match="max_length must be > 0"):
        get_scaled_t_disrupt(1002, str(tmp_path), 0.5, 0)


def test_get_scaled_t_disrupt_returns_closest_time_index_fraction(tmp_path: Path) -> None:
    """Map disruption time to the nearest timestep index fraction of ``max_length``."""
    write_signal_file(tmp_path / "1003.txt", [(0.0, 1.0), (0.4, 2.0), (0.9, 3.0)])

    scaled = get_scaled_t_disrupt(1003, str(tmp_path), 0.5, 10)

    assert scaled == pytest.approx(0.1)


def test_load_and_pad_variants_pad_to_requested_length(tmp_path: Path) -> None:
    """Load, normalize, and pad raw signals to the requested fixed-length output."""
    write_signal_file(tmp_path / "1004.txt", [(0.0, 2.0), (1.0, 4.0), (2.0, 6.0)])

    shot_no, raw = load_and_pad("1004.txt", str(tmp_path), 5)
    _, norm = load_and_pad_norm("1004.txt", str(tmp_path), 5, mean=4.0, std=2.0)
    _, scaled = load_and_pad_scale("1004.txt", str(tmp_path), 5)

    assert shot_no == 1004
    assert raw.tolist() == pytest.approx([2.0, 4.0, 6.0, 0.0, 0.0])
    assert norm.tolist() == pytest.approx([-1.0, 0.0, 1.0, 0.0, 0.0])
    assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.0, 0.0])
    assert raw.dtype == np.float32
