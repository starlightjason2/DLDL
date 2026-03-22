from pathlib import Path
import types

import numpy as np
import pytest

from helpers import load_module_from_path


def load_dataset_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Load the dataset module with temp env paths and lightweight fake dependencies."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("1001 -1.0\n1002 0.4\n", encoding="utf-8")
    data_path = tmp_path / "processed" / "dataset.pt"
    labels_pt_path = tmp_path / "processed" / "labels.pt"

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("LABELS_PATH", str(labels_path))
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setenv("TRAIN_LABELS_PATH", str(labels_pt_path))
    monkeypatch.setenv("NORMALIZATION_TYPE", "meanvar-whole")
    monkeypatch.setenv("CPU_USE", "0.25")
    monkeypatch.setenv("PREPROCESSOR_MAX_WORKERS", "2")

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = np.ndarray
    fake_torch_store: dict[str, np.ndarray] = {}
    fake_torch.tensor = lambda value: np.array(value)
    fake_torch.save = lambda value, path: fake_torch_store.__setitem__(
        str(path), np.array(value)
    )
    fake_torch.load = lambda path: np.array(fake_torch_store[str(path)])

    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    fake_torch_utils_data.Dataset = object

    processing_module = types.ModuleType("util.processing")
    processing_module.convert_tensors_to_float = lambda *_args: None
    processing_module.create_binary_labels = lambda shot_list: np.column_stack(
        ((shot_list[:, 1] != -1.0).astype(float), shot_list[:, 1])
    )
    processing_module.get_use_cores = lambda cpu_use: max(1, int(cpu_use * 8))

    data_loading_module = load_module_from_path(
        "test_dataset_data_loading",
        Path(__file__).resolve().parents[1] / "src" / "util" / "data_loading.py",
    )
    util_package = types.ModuleType("util")
    util_package.data_loading = data_loading_module
    util_package.processing = processing_module

    module = load_module_from_path(
        "test_model_dataset",
        Path(__file__).resolve().parents[1] / "src" / "model" / "dataset.py",
        injected_modules={
            "torch": fake_torch,
            "torch.utils": fake_torch_utils,
            "torch.utils.data": fake_torch_utils_data,
            "util": util_package,
            "util.data_loading": data_loading_module,
            "util.processing": processing_module,
        },
    )
    return module, fake_torch_store, data_dir, labels_path, data_path, labels_pt_path


def test_ipdataset_loads_existing_saved_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load existing saved dataset tensors and expose samples through ``__getitem__``."""
    module, store, _data_dir, _labels_path, data_path, labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("", encoding="utf-8")
    labels_pt_path.write_text("", encoding="utf-8")
    store[str(data_path)] = np.array([[1.0, 2.0], [3.0, 4.0]])
    store[str(labels_pt_path)] = np.array([[0.0, -1.0], [1.0, 0.2]])

    dataset = module.IpDataset(normalization_type="scale")
    cls_dataset = module.IpDataset(normalization_type="scale", classification=True)

    assert len(dataset) == 2
    assert dataset[1][0].tolist() == [3.0, 4.0]
    assert dataset[1][1].tolist() == [1.0, 0.2]
    assert cls_dataset[1][1] == 1.0


def test_ipdataset_missing_files_builds_state_before_loading_saved_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialize preprocessing state and invoke dataset building when cached tensors are missing."""
    module, store, _data_dir, _labels_path, data_path, labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    called = {"make_dataset": False}

    monkeypatch.setattr(module.IpDataset, "_get_max_length", lambda self: 7)
    monkeypatch.setattr(module.IpDataset, "_get_mean_std", lambda self: (1.5, 0.5))

    def fake_make_dataset(self, make_labels: bool = True) -> None:
        called["make_dataset"] = make_labels
        Path(self.data_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.data_file).write_text("", encoding="utf-8")
        Path(self.labels_file).write_text("", encoding="utf-8")
        store[str(self.data_file)] = np.array([[10.0, 11.0]])
        store[str(self.labels_file)] = np.array([[1.0, 0.4]])

    monkeypatch.setattr(module.IpDataset, "_make_dataset", fake_make_dataset)

    dataset = module.IpDataset(normalization_type="meanvar-whole")

    assert called["make_dataset"] is True
    assert dataset.max_length == 7
    assert dataset.mean == 1.5
    assert dataset.std == 0.5
    assert dataset.data.tolist() == [[10.0, 11.0]]
    assert dataset.labels.tolist() == [[1.0, 0.4]]


def test_load_single_file_dispatches_by_normalization_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch a raw-shot load to the normalization-specific helper function."""
    module, _store, _data_dir, _labels_path, _data_path, _labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    dataset = module.IpDataset.__new__(module.IpDataset)
    dataset.max_length = 5
    dataset.mean = 2.0
    dataset.std = 4.0
    dataset.normalization_type = "meanvar-whole"

    monkeypatch.setattr(
        module,
        "load_and_pad_norm",
        lambda filename, data_dir, max_length, mean, std: (
            int(filename[:-4]),
            np.array([max_length, mean, std], dtype=np.float32),
        ),
    )

    shot_no, values = dataset._load_single_file("1002.txt")

    assert shot_no == 1002
    assert values.tolist() == pytest.approx([5.0, 2.0, 4.0])


def test_load_single_file_rejects_unknown_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise when a dataset is asked to use an unsupported normalization mode."""
    module, _store, _data_dir, _labels_path, _data_path, _labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    dataset = module.IpDataset.__new__(module.IpDataset)
    dataset.max_length = 5
    dataset.mean = None
    dataset.std = None
    dataset.normalization_type = "unknown"

    with pytest.raises(ValueError, match="Unknown normalization"):
        dataset._load_single_file("1002.txt")


def test_create_labels_in_sorted_order_scales_only_disruptive_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scale time labels only for disruptive shots when building sorted labels."""
    module, _store, _data_dir, _labels_path, _data_path, _labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    dataset = module.IpDataset.__new__(module.IpDataset)
    dataset.labels_type = "scaled"
    dataset.sorted_shot_numbers = np.array([1001, 1002])
    dataset._shot_to_label = {
        1001: np.array([1001.0, -1.0]),
        1002: np.array([1002.0, 0.4]),
    }
    dataset.max_length = 9

    monkeypatch.setattr(module, "get_scaled_t_disrupt", lambda *_args: 0.75)

    labels = dataset._create_labels_in_sorted_order()

    assert np.allclose(labels, np.array([[0.0, -1.0], [1.0, 0.75]]))


def test_make_labels_can_save_binary_and_scaled_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create labels from the shot list and persist them through ``torch.save`` when requested."""
    module, store, _data_dir, _labels_path, _data_path, labels_pt_path = load_dataset_module(
        tmp_path, monkeypatch
    )
    dataset = module.IpDataset.__new__(module.IpDataset)
    dataset.shot_list = np.array([[1001.0, -1.0], [1002.0, 0.4]])
    dataset.max_length = 10
    dataset.labels_file = str(labels_pt_path)

    monkeypatch.setattr(module, "get_scaled_t_disrupt", lambda *_args: 0.3)

    labels = dataset.make_labels(scaled=True, save=True)

    expected = np.array([[0.0, -1.0], [1.0, 0.3]])
    assert np.allclose(labels, expected)
    assert np.allclose(store[str(labels_pt_path)], expected)
