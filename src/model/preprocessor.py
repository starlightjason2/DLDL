"""Data preprocessing utilities for plasma disruption datasets."""

import os
import random

from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from numpy import float32
from numpy.typing import NDArray
from torch import Tensor

from constants import CPU_USE, DATA_DIR, LABELS_PATH, NORMALIZATION_TYPE
from model.model import IpDataset
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
    get_processed_dataset_path,
    get_processed_labels_path,
    get_use_cores,
)


class Preprocessor:
    """Preprocessor for plasma current time series data."""

    def __init__(
        self,
        dataset_id: str = "",
        dset_path: Optional[str] = None,
        labels_path: Optional[str] = None,
    ) -> None:
        """Initialize preprocessor.

        Uses NORMALIZATION_TYPE and CPU_USE from constants (env). Override paths via args.

        Args:
            dataset_id: Optional identifier for output filenames. Defaults to NORMALIZATION_TYPE.
            dset_path: Optional custom dataset path. Defaults to processed_dataset_{dataset_id}.pt.
            labels_path: Optional custom labels path. Defaults to processed_labels_{dataset_id}.pt.
        """
        self.dataset_id = dataset_id or NORMALIZATION_TYPE
        self.cpu_use = CPU_USE
        self.normalization = NORMALIZATION_TYPE
        self.dset_path = dset_path or get_processed_dataset_path(self.dataset_id)
        self.labels_path = labels_path or get_processed_labels_path(self.dataset_id)
        self.logger = logger.bind(name=__name__)

        self.shot_list = np.loadtxt(LABELS_PATH)
        self.file_list = [f"{int(shot[0])}.txt" for shot in self.shot_list]
        self.num_shots = len(self.file_list)
        self.max_length = self._get_max_length()
        self.mean, self.std = (
            self._get_mean_std()
            if NORMALIZATION_TYPE == "meanvar-whole"
            else (None, None)
        )
        self.sorted_shot_numbers: Optional[NDArray] = None

        self._shot_to_idx = {
            int(shot[0]): idx for idx, shot in enumerate(self.shot_list)
        }
        self._shot_to_label = {int(shot[0]): shot.copy() for shot in self.shot_list}

        if not os.path.exists(self.dset_path) or not os.path.exists(self.labels_path):
            self.logger.info(
                f"Creating new dataset (dataset_id='{self.dataset_id}', normalization={self.normalization})"
            )
            self._make_dataset(make_labels=True, labels_type="scaled")
        else:
            self.logger.info(f"Loading existing dataset from {self.dset_path}")
            self._load_sorted_shot_numbers()

    def _load_sorted_shot_numbers(self) -> None:
        """Load sorted shot numbers from shotlist."""
        self.sorted_shot_numbers = np.sort(self.shot_list[:, 0].astype(int))

    def _load_single_file(self, filename: str) -> Tuple[int, NDArray[float32]]:
        """Load and preprocess a single file based on normalization setting.

        Args:
            filename: Name of the signal file (e.g., "12345.txt").

        Returns:
            Tuple of (shot_number, preprocessed_data).
        """
        loaders = {
            "scale": lambda: load_and_pad_scale(filename, DATA_DIR, self.max_length),
            "meanvar-whole": lambda: load_and_pad_norm(
                filename, DATA_DIR, self.max_length, self.mean, self.std
            ),
            "meanvar-single": lambda: load_and_pad_norm(
                filename, DATA_DIR, self.max_length, None, None
            ),
        }
        if self.normalization not in loaders:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return loaders[self.normalization]()

    def _process_files_parallel(self, func: Callable[..., Any], *args: Any) -> NDArray:
        """Process files in parallel using ProcessPoolExecutor.

        Args:
            func: Function to apply to each file.
            *args: Additional arguments to pass to func.

        Returns:
            Array of results from parallel processing.
        """
        use_cores = get_use_cores(self.cpu_use)
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            return np.asarray(
                list(
                    executor.map(
                        func, self.file_list, [DATA_DIR] * self.num_shots, *args
                    )
                )
            )

    def _get_max_length(self) -> int:
        """Compute maximum time series length across all shots."""
        max_len = int(np.max(self._process_files_parallel(get_length)))
        self.logger.info(f"Maximum time series length: {max_len} timesteps")
        return max_len

    def _get_mean_std(self) -> Tuple[float, float]:
        """Compute dataset-wide mean and std. Returns (mean, std) using std = sqrt(E[X^2] - E[X]^2)."""
        results = self._process_files_parallel(get_means)
        mean = float(np.mean(results[:, 0]))
        variance = max(0.0, float(np.mean(results[:, 1]) - mean**2))  # Guard against numerical errors
        std = float(variance ** 0.5)
        self.logger.info(f"Dataset statistics: mean={mean:.6f}, std={std:.6f}")
        return mean, std

    def make_labels_naive(self, save: bool = False) -> NDArray[np.float64]:
        """Create binary classification labels.

        Args:
            save: If True, save labels to file.

        Returns:
            Labels array (n_shots, 2): [binary_class, original_time].
        """
        labels = create_binary_labels(self.shot_list)
        if save:
            torch.save(torch.tensor(labels), self.labels_path)
        return labels

    def make_labels_scaled(self, save: bool = False) -> NDArray[np.float64]:
        """Create labels with binary classification and scaled disruption time.

        Args:
            save: If True, save labels to file.

        Returns:
            Labels array (n_shots, 2): [binary_class, scaled_time].
        """
        labels = create_binary_labels(self.shot_list)
        for label_row, shot_row in zip(labels, self.shot_list):
            if shot_row[1] != -1.0:
                label_row[1] = get_scaled_t_disrupt(
                    int(shot_row[0]), DATA_DIR, shot_row[1], self.max_length
                )
        if save:
            torch.save(torch.tensor(labels), self.labels_path)
        return labels

    def _get_normalization_loader(
        self,
        data_dir_args: List[str],
        max_length_args: List[int],
    ) -> Tuple[Callable[..., Tuple[int, NDArray[float32]]], Tuple[Any, ...]]:
        """Get loader function and arguments for current normalization method.

        Args:
            data_dir_args: List of data directory paths (one per shot).
            max_length_args: List of max_length values (one per shot).

        Returns:
            Tuple of (loader_function, loader_arguments) for executor.map().
        """
        loaders: Dict[
            Literal["scale", "meanvar-whole", "meanvar-single"],
            Tuple[Callable[..., Tuple[int, NDArray[float32]]], Tuple[Any, ...]],
        ] = {
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
        if self.normalization not in loaders:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return loaders[self.normalization]

    def _make_dataset(
        self,
        make_labels: bool = True,
        labels_type: Literal["scaled", "naive"] = "scaled",
    ) -> None:
        """Build preprocessed dataset from raw signal files.

        Args:
            make_labels: If True, create and save labels tensor.
            labels_type: Label type ('scaled' or 'naive').
        """
        self.logger.info(
            f"Building dataset for {self.num_shots} shots (normalization={self.normalization})"
        )
        self.logger.info(
            f"Estimated memory: ~{(self.num_shots * self.max_length * 4) / (1024**3):.2f} GB"
        )

        loader_func, loader_args = self._get_normalization_loader(
            [DATA_DIR] * self.num_shots, [self.max_length] * self.num_shots
        )

        with ProcessPoolExecutor(max_workers=get_use_cores(self.cpu_use)) as executor:
            results = list[Tuple[int, NDArray[float32]]](
                executor.map(loader_func, *loader_args)
            )

        sorted_data = sorted(results, key=lambda x: x[0])
        sorted_shot_numbers, dataset_data = zip(*sorted_data)
        self.sorted_shot_numbers = np.array(sorted_shot_numbers)

        torch.save(torch.tensor(np.array(dataset_data)), self.dset_path)
        self.logger.info(f"Saved dataset: {self.dset_path}")

        if make_labels:
            torch.save(
                self._create_labels_in_sorted_order(sorted_shot_numbers, labels_type),
                self.labels_path,
            )
            self.logger.info(f"Saved labels: {self.labels_path}")

        self.logger.info("Converting to float32...")
        convert_tensors_to_float(self.dset_path, self.labels_path)

    def _create_labels_in_sorted_order(
        self,
        sorted_shot_numbers: NDArray,
        labels_type: Literal["scaled", "naive"],
    ) -> Tensor:
        """Create labels in sorted shot number order.

        Args:
            sorted_shot_numbers: Array of shot numbers in sorted order.
            labels_type: Label type ('scaled' or 'naive').

        Returns:
            Labels tensor matching sorted shot order.
        """
        labels = np.array(
            [self._shot_to_label[int(shot)] for shot in sorted_shot_numbers]
        )
        if labels_type == "scaled":
            for label_row, shot_no in zip(labels, sorted_shot_numbers):
                t_disrupt = label_row[1]
                if t_disrupt != -1.0:
                    label_row[0], label_row[1] = 1, get_scaled_t_disrupt(
                        int(shot_no), DATA_DIR, t_disrupt, self.max_length
                    )
                else:
                    label_row[0] = 0
        else:
            labels[:, 0] = (labels[:, 1] != -1.0).astype(float)
        return torch.tensor(labels)

    def _load_example_from_raw(
        self, idx: int, scale_labels: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Load single example from raw files.

        Args:
            idx: Index in sorted dataset.
            scale_labels: If True, scale disruption time to [0, 1].

        Returns:
            Tuple of (data_tensor, label_tensor).
        """
        shot_no = int(self.sorted_shot_numbers[idx])
        t_disrupt = self.shot_list[self._shot_to_idx[shot_no], 1]
        is_disruptive = t_disrupt != -1.0
        label = np.array([float(is_disruptive), t_disrupt if is_disruptive else -1.0])

        if is_disruptive and scale_labels:
            label[1] = get_scaled_t_disrupt(
                shot_no, DATA_DIR, t_disrupt, self.max_length
            )

        return torch.tensor(self._load_single_file(f"{shot_no}.txt")[1]), torch.tensor(
            label
        )

    def check_dataset(
        self, scale_labels: bool = True, num_checks: int = 100, verbose: bool = False
    ) -> None:
        """Verify preprocessed dataset integrity by comparing with raw file processing.

        Args:
            scale_labels: Whether labels were scaled during preprocessing.
            num_checks: Number of random examples to verify.
            verbose: If True, log each verification step.
        """
        dataset = IpDataset(self.dset_path, self.labels_path)
        num_checks = min(num_checks, len(dataset))  # Guard against num_checks > dataset size
        self.logger.info(
            f"Verifying dataset integrity: {num_checks} examples (scale_labels={scale_labels})"
        )

        for idx in random.sample(range(len(dataset)), num_checks):
            shot_no = int(self.sorted_shot_numbers[idx])
            if verbose:
                self.logger.debug(f"Verifying shot {shot_no} at index {idx}")

            proc_data, proc_label = dataset[idx]
            exp_data, exp_label = self._load_example_from_raw(idx, scale_labels)

            proc_data, proc_label = proc_data.squeeze(0), proc_label.squeeze(0)
            # Cast expected to proc dtype (Float vs Double) then use approximate equality
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
