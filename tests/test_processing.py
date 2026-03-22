import types
from pathlib import Path

import numpy as np

from helpers import load_module_from_path


class FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value

    def float(self) -> str:
        return f"{self.value}-float"


class FakeSubset:
    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)


def load_processing_module():
    fake_torch = types.ModuleType("torch")
    saved = {}
    loaded = {
        "dataset.pt": FakeTensor("dataset"),
        "labels.pt": FakeTensor("labels"),
    }
    fake_torch.load = lambda path: loaded[path]
    fake_torch.save = lambda value, path: saved.__setitem__(path, value)

    fake_utils = types.ModuleType("torch.utils")
    fake_utils_data = types.ModuleType("torch.utils.data")
    fake_utils_data.Dataset = object
    fake_utils_data.Subset = FakeSubset

    module = load_module_from_path(
        "test_util_processing",
        Path(__file__).resolve().parents[1] / "src" / "util" / "processing.py",
        injected_modules={
            "torch": fake_torch,
            "torch.utils": fake_utils,
            "torch.utils.data": fake_utils_data,
        },
    )
    return module, saved


def test_get_use_cores_and_create_binary_labels(monkeypatch) -> None:
    """Derive worker counts and binary disruption labels from simple numeric inputs."""
    module, _ = load_processing_module()
    monkeypatch.setattr(module.mp, "cpu_count", lambda: 16)

    assert module.get_use_cores(0.25) == 4

    labels = module.create_binary_labels(np.array([[1, -1.0], [2, 0.3], [3, -1.0]]))
    assert labels[:, 0].tolist() == [0.0, 1.0, 0.0]


def test_convert_tensors_to_float_uses_torch_load_and_save() -> None:
    """Reload tensors, cast them to float, and save them back to disk paths."""
    module, saved = load_processing_module()

    module.convert_tensors_to_float("dataset.pt", "labels.pt")

    assert saved["dataset.pt"] == "dataset-float"
    assert saved["labels.pt"] == "labels-float"


def test_split_returns_train_dev_test_subsets() -> None:
    """Split a dataset into train, dev, and test subsets using index ranges."""
    module, _ = load_processing_module()
    dataset = list(range(10))

    train, dev, test = module.split(dataset, train_size=0.6)

    assert train.indices == list(range(0, 6))
    assert dev.indices == list(range(6, 8))
    assert test.indices == list(range(8, 10))
