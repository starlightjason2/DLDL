from pathlib import Path
import types

import pytest

from helpers import load_module_from_path


def load_cnn_module():
    """Load the CNN module with fake torch-style dependencies for core unit tests."""
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.cuda = types.SimpleNamespace(
        device_count=lambda: 0,
        is_available=lambda: False,
        set_device=lambda *_args: None,
    )
    fake_torch.manual_seed = lambda *_args: None
    fake_torch.zeros = lambda *_args, **_kwargs: types.SimpleNamespace(
        numel=lambda: 4, size=lambda _idx: 1
    )

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    fake_torch.no_grad = lambda: FakeNoGrad()

    fake_dist = types.ModuleType("torch.distributed")
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = object
    fake_nn.BCEWithLogitsLoss = lambda: None
    fake_nn.MSELoss = lambda: None
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_optim = types.ModuleType("torch.optim")
    fake_nn_parallel = types.ModuleType("torch.nn.parallel")
    fake_nn_parallel.DistributedDataParallel = object
    fake_utils = types.ModuleType("torch.utils")
    fake_utils_data = types.ModuleType("torch.utils.data")
    fake_utils_data.DataLoader = object
    fake_utils_data.DistributedSampler = object
    fake_utils_tensorboard = types.ModuleType("torch.utils.tensorboard")
    fake_utils_tensorboard.SummaryWriter = object

    fake_settings = types.ModuleType("config.settings")
    fake_settings.load_settings = lambda: types.SimpleNamespace(
        training_config=types.SimpleNamespace(
            learning_rate=0.001,
            num_epochs=1,
            log_interval=1,
            weight_decay=0.0,
            lr_scheduler=True,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1,
            early_stopping_patience=1,
            gradient_clip=1.0,
            batch_size=2,
        )
    )
    fake_util_distributed = types.ModuleType("util.distributed")
    fake_util_distributed.cleanup = lambda: None
    fake_util_distributed.setup = lambda *_args: None
    fake_util_processing = types.ModuleType("util.processing")
    fake_util_processing.split = lambda dataset: (dataset, dataset, dataset)
    fake_model_dataset = types.ModuleType("model.dataset")
    fake_model_dataset.IpDataset = object

    return load_module_from_path(
        "test_model_cnn",
        Path(__file__).resolve().parents[1] / "src" / "model" / "cnn.py",
        injected_modules={
            "torch": fake_torch,
            "torch.distributed": fake_dist,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_functional,
            "torch.optim": fake_optim,
            "torch.nn.parallel": fake_nn_parallel,
            "torch.utils": fake_utils,
            "torch.utils.data": fake_utils_data,
            "torch.utils.tensorboard": fake_utils_tensorboard,
            "config.settings": fake_settings,
            "util.distributed": fake_util_distributed,
            "util.processing": fake_util_processing,
            "model.dataset": fake_model_dataset,
        },
    )


def test_validate_preprocessed_files_raises_for_missing_artifacts(
    tmp_path: Path,
) -> None:
    """Raise a clear error when the expected preprocessed dataset files are missing."""
    module = load_cnn_module()
    model = module.IpCNN.__new__(module.IpCNN)
    errors: list[str] = []
    model.data_path = str(tmp_path / "missing_dataset.pt")
    model.labels_path = str(tmp_path / "missing_labels.pt")
    model.logger = types.SimpleNamespace(error=lambda message: errors.append(message))

    with pytest.raises(FileNotFoundError, match="Run preprocess_data.py first"):
        model.validate_preprocessed_files("scale")

    assert "Preprocessed files not found" in errors[0]


def test_len_sums_parameter_sizes_without_building_full_model() -> None:
    """Compute model length as the sum of ``numel`` across all parameters."""
    module = load_cnn_module()
    model = module.IpCNN.__new__(module.IpCNN)
    model.parameters = lambda: [
        types.SimpleNamespace(numel=lambda: 3),
        types.SimpleNamespace(numel=lambda: 5),
        types.SimpleNamespace(numel=lambda: 7),
    ]

    assert len(model) == 15
