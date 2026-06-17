"""Dataset and preprocessing utilities."""

from dataclasses import dataclass
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Literal, Sized, Tuple, cast

import numpy as np
import torch
from loguru import logger
from numpy import float32
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset, Subset

from util.data_loading import (
    get_length,
    get_means,
    get_scaled_t_disrupt,
    load_and_pad_norm,
    load_and_pad_scale,
)
from util.processing import (
    convert_tensors_to_float,
    create_binary_labels,
    get_use_cores,
)


@dataclass(frozen=True)
class ShotView:
    index: int
    shot_no: int
    time: np.ndarray
    current: np.ndarray
    disruptive: bool
    t_disrupt: float | None
    normalization_type: str

    @property
    def title(self) -> str:
        status = "disruptive" if self.disruptive else "non-disruptive"
        return (
            f"Shot {self.shot_no} (index {self.index}, {status}, "
            f"{self.normalization_type})"
        )


class IpDataset(Dataset):
    """PyTorch Dataset for plasma current time series data."""

    def __init__(
        self,
        normalization_type: str,
        data_file: str,
        labels_file: str,
        labels_path: str,
        data_dir: str,
        labels_type: Literal["scaled", "naive"],
        cpu_use: float,
        preprocessor_max_workers: int,
    ) -> None:
        """Initialize dataset, creating preprocessed files if missing."""
        self.normalization_type = normalization_type
        self.data_file = data_file
        self.labels_file = labels_file
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.labels_type = labels_type
        self.cpu_use = cpu_use
        self.preprocessor_max_workers = preprocessor_max_workers
        self.logger = logger.bind(name=__name__)

        if not os.path.exists(self.data_file) or not os.path.exists(self.labels_file):
            self.logger.info(
                f"Preprocessed files not found. Creating dataset "
                f"(normalization_type='{self.normalization_type}')"
            )
            self.shot_list = np.loadtxt(self.labels_path)
            self.file_list = [f"{int(shot[0])}.txt" for shot in self.shot_list]
            self.num_shots = len(self.file_list)
            self.max_length = self._get_max_length()
            self.mean, self.std = (
                self._get_mean_std()
                if self.normalization_type == "meanvar-whole"
                else (None, None)
            )
            self.sorted_shot_numbers = None
            self._shot_to_idx = {
                int(shot[0]): idx for idx, shot in enumerate(self.shot_list)
            }
            self._shot_to_label = {int(shot[0]): shot.copy() for shot in self.shot_list}
            self.sorted_shot_numbers = np.sort(self.shot_list[:, 0].astype(int))
            self._make_dataset(make_labels=True)
        else:
            self.logger.info(f"Loading existing dataset from {self.data_file}")

        self.data = torch.load(self.data_file, mmap=True, weights_only=True)
        self.labels = torch.load(self.labels_file, mmap=True, weights_only=True)
        self.logger.info(f"Ready: {len(self.data)} shots")
        self._ensure_shot_metadata()

    def _ensure_shot_metadata(self) -> None:
        """Load shot-number index tables when reading cached ``.pt`` files."""
        if getattr(self, "sorted_shot_numbers", None) is not None:
            return
        self.shot_list = np.loadtxt(self.labels_path)
        if self.shot_list.ndim == 1:
            self.shot_list = self.shot_list.reshape(1, -1)
        self.sorted_shot_numbers = np.sort(self.shot_list[:, 0].astype(int))

    def shot_number(self, idx: int) -> int:
        """Return the shot id at ``idx`` (sorted-shot order used by ``__getitem__``)."""
        self._ensure_shot_metadata()
        return int(self.sorted_shot_numbers[idx])

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get sample and label at index."""
        return self.data[idx], self.labels[idx]

    def _load_single_file(self, filename: str) -> Tuple[int, NDArray[float32]]:
        """Load and preprocess a single file."""
        loaders = {
            "scale": lambda: load_and_pad_scale(
                filename, self.data_dir, self.max_length
            ),
            "meanvar-whole": lambda: load_and_pad_norm(
                filename, self.data_dir, self.max_length, self.mean, self.std
            ),
            "meanvar-single": lambda: load_and_pad_norm(
                filename, self.data_dir, self.max_length, None, None
            ),
        }
        if self.normalization_type not in loaders:
            raise ValueError(f"Unknown normalization: {self.normalization_type}")
        return loaders[self.normalization_type]()

    def _process_files_parallel(
        self, func: Callable[..., Any], *args: Any, desc: str = "Processing"
    ) -> NDArray:
        """Process files in parallel."""
        workers = min(get_use_cores(self.cpu_use), self.preprocessor_max_workers)
        chunksize = max(1, self.num_shots // (workers * 4))  # Reduce IPC overhead
        self.logger.info(f"{desc}: {self.num_shots} files, {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            it = executor.map(
                func,
                self.file_list,
                [self.data_dir] * self.num_shots,
                *args,
                chunksize=chunksize,
            )
            return np.asarray(list(it))

    def _get_max_length(self) -> int:
        """Compute maximum time series length across all shots."""
        max_len = int(
            np.max(self._process_files_parallel(get_length, desc="Getting max length"))
        )
        self.logger.info(f"Maximum time series length: {max_len} timesteps")
        return max_len

    def _get_mean_std(self) -> Tuple[float, float]:
        """Compute dataset-wide mean and std."""
        results = self._process_files_parallel(get_means, desc="Computing mean/std")
        mean = float(np.mean(results[:, 0]))
        std = float(max(0.0, float(np.mean(results[:, 1]) - mean**2)) ** 0.5)
        self.logger.info(f"Dataset statistics: mean={mean:.6f}, std={std:.6f}")
        return mean, std

    def load_shot_view(self, index: int) -> ShotView:
        shot_no = self.shot_number(index)
        signal, label = self[index]
        current = signal.squeeze().numpy()
        label = label.squeeze().numpy()
        disruptive = bool(label[0] > 0.5)

        return ShotView(
            index=index,
            shot_no=shot_no,
            time=np.arange(len(current)) / len(current),
            current=current,
            disruptive=disruptive,
            t_disrupt=float(label[1]) if disruptive else None,
            normalization_type=self.normalization_type,
        )

    def make_labels(
        self, scaled: bool = True, save: bool = False
    ) -> NDArray[np.float64]:
        """Create labels with optional scaling."""
        labels = create_binary_labels(self.shot_list)
        if scaled:
            for label_row, shot_row in zip(labels, self.shot_list):
                if shot_row[1] != -1.0:
                    label_row[1] = get_scaled_t_disrupt(
                        int(shot_row[0]), self.data_dir, shot_row[1], self.max_length
                    )
        if save:
            torch.save(torch.tensor(labels), self.labels_file)
        return labels

    def _get_normalization_loader(
        self, data_dir_args: List[str], max_length_args: List[int]
    ) -> Tuple[Callable[..., Tuple[int, NDArray[float32]]], Tuple[Any, ...]]:
        """Get loader function and arguments for current normalization method."""
        loaders = {
            "scale": (
                load_and_pad_scale,
                (self.file_list, data_dir_args, max_length_args),
            ),
            "meanvar-whole": (
                load_and_pad_norm,
                (
                    self.file_list,
                    data_dir_args,
                    max_length_args,
                    [self.mean] * self.num_shots,
                    [self.std] * self.num_shots,
                ),
            ),
            "meanvar-single": (
                load_and_pad_norm,
                (
                    self.file_list,
                    data_dir_args,
                    max_length_args,
                    [None] * self.num_shots,
                    [None] * self.num_shots,
                ),
            ),
        }
        if self.normalization_type not in loaders:
            raise ValueError(f"Unknown normalization: {self.normalization_type}")
        return loaders[self.normalization_type]

    def _make_dataset(self, make_labels: bool = True) -> None:
        """Build preprocessed dataset from raw signal files."""
        self.logger.info(
            f"Building dataset for {self.num_shots} shots (normalization={self.normalization_type})"
        )
        self.logger.info(
            f"Estimated memory: ~{(self.num_shots * self.max_length * 4) / (1024**3):.2f} GB"
        )

        loader_func, loader_args = self._get_normalization_loader(
            [self.data_dir] * self.num_shots, [self.max_length] * self.num_shots
        )

        workers = min(get_use_cores(self.cpu_use), self.preprocessor_max_workers)
        chunksize = max(1, self.num_shots // (workers * 4))  # Reduce IPC overhead
        self.logger.info(
            f"Loading and normalizing: {self.num_shots} files, {workers} workers"
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(loader_func, *loader_args, chunksize=chunksize))

        sorted_data = sorted(results, key=lambda x: x[0])
        sorted_shot_numbers, dataset_data = zip(*sorted_data)
        self.sorted_shot_numbers = np.array(sorted_shot_numbers)

        torch.save(torch.tensor(np.array(dataset_data)), self.data_file)
        self.logger.info(f"Saved dataset: {self.data_file}")

        if make_labels:
            torch.save(
                self._create_labels_in_sorted_order(),
                self.labels_file,
            )
            self.logger.info(f"Saved labels: {self.labels_file}")

        self.logger.info("Converting to float32...")
        convert_tensors_to_float(self.data_file, self.labels_file)

    def _create_labels_in_sorted_order(self) -> Tensor:
        """Create labels in sorted shot number order."""
        labels = np.array(
            [self._shot_to_label[int(shot)] for shot in self.sorted_shot_numbers]
        )
        if self.labels_type == "scaled":
            for label_row, shot_no in zip(labels, self.sorted_shot_numbers):
                t_disrupt = label_row[1]
                if t_disrupt != -1.0:
                    label_row[0], label_row[1] = 1, get_scaled_t_disrupt(
                        int(shot_no), self.data_dir, t_disrupt, self.max_length
                    )
                else:
                    label_row[0] = 0
        else:
            labels[:, 0] = (labels[:, 1] != -1.0).astype(float)
        return torch.tensor(labels)

    def _load_example_from_raw(
        self, idx: int, scale_labels: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Load single example from raw files."""
        shot_no = int(self.sorted_shot_numbers[idx])
        t_disrupt = self.shot_list[self._shot_to_idx[shot_no], 1]
        is_disruptive = t_disrupt != -1.0
        label = np.array([float(is_disruptive), t_disrupt if is_disruptive else -1.0])

        if is_disruptive and scale_labels:
            label[1] = get_scaled_t_disrupt(
                shot_no, self.data_dir, t_disrupt, self.max_length
            )

        return torch.tensor(self._load_single_file(f"{shot_no}.txt")[1]), torch.tensor(
            label
        )

    def check_dataset(
        self, scale_labels: bool = True, num_checks: int = 100, verbose: bool = False
    ) -> None:
        """Verify preprocessed dataset integrity."""
        # Initialize preprocessing state if not already done
        if not hasattr(self, "shot_list"):
            self.shot_list = np.loadtxt(self.labels_path)
            self.file_list = [f"{int(shot[0])}.txt" for shot in self.shot_list]
            self.num_shots = len(self.file_list)
            self.max_length = self._get_max_length()
            self.mean, self.std = (
                self._get_mean_std()
                if self.normalization_type == "meanvar-whole"
                else (None, None)
            )
            self.sorted_shot_numbers = None
            self._shot_to_idx = {
                int(shot[0]): idx for idx, shot in enumerate(self.shot_list)
            }
            self._shot_to_label = {int(shot[0]): shot.copy() for shot in self.shot_list}
            self.sorted_shot_numbers = np.sort(self.shot_list[:, 0].astype(int))

        # Reload data to ensure we're checking the saved files
        data = torch.load(self.data_file)
        labels = torch.load(self.labels_file)

        num_checks = min(num_checks, len(data))
        self.logger.info(
            f"Verifying dataset integrity: {num_checks} examples (scale_labels={scale_labels})"
        )

        for idx in random.sample(range(len(data)), num_checks):
            shot_no = int(self.sorted_shot_numbers[idx])
            if verbose:
                self.logger.debug(f"Verifying shot {shot_no} at index {idx}")

            proc_data = data[idx]
            proc_label = labels[idx]
            exp_data, exp_label = self._load_example_from_raw(idx, scale_labels)

            proc_data, proc_label = proc_data.squeeze(0), proc_label.squeeze(0)
            exp_data = exp_data.to(proc_data.dtype)
            exp_label = exp_label.to(proc_label.dtype)
            data_match = torch.allclose(proc_data, exp_data, rtol=1e-5, atol=1e-8)
            label_match = torch.allclose(proc_label, exp_label, rtol=1e-5, atol=1e-8)

            if not (data_match and label_match):
                self.logger.warning(f"Mismatch at index {idx} (shot {shot_no})")
                if not data_match:
                    max_diff = torch.max(torch.abs(proc_data - exp_data)).item()
                    self.logger.warning(f"  Data diff: {max_diff:.9f}")
                if not label_match:
                    max_label_diff = torch.max(torch.abs(proc_label - exp_label)).item()
                    self.logger.warning(
                        f"  Label diff: {max_label_diff:.9f} "
                        f"({proc_label.tolist()} vs {exp_label.tolist()})"
                    )
                self.logger.error("Dataset integrity check failed")
                return

        self.logger.info(f"Integrity check passed: all {num_checks} examples match")

    def split(self, train_size: float = 0.8):
        """Split dataset into train, dev, and test sets."""
        total_size = len(cast(Sized, self))
        train_end = int(train_size * total_size)
        dev_end = int((train_size + (1 - train_size) / 2) * total_size)
        return (
            Subset(self, range(0, train_end)),
            Subset(self, range(train_end, dev_end)),
            Subset(self, range(dev_end, total_size)),
        )
