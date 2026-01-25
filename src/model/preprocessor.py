"""Data preprocessing utilities for plasma disruption datasets."""

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from numpy import float32
from numpy.typing import NDArray
from torch import Tensor

from constants import DATA_DIR, LABELS_PATH
from model.model import IpDataset
from util.data_loading import (
    get_length,
    get_means,
    get_scaled_t_disrupt,
    load_and_pad,
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
        cpu_use: float,
        dataset_id: str = "",
        normalization: Optional[
            Literal["scale", "meanvar-whole", "meanvar-single"]
        ] = None,
        dset_path: Optional[str] = None,
        labels_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            dataset_id: Optional identifier appended to output filenames.
            cpu_use: Fraction of CPU cores for parallel processing (0.0 to 1.0).
            normalization: Normalization strategy (None, 'scale', 'meanvar-whole', or 'meanvar-single').
            dset_path: Path to dataset file. If None, uses default path.
            labels_path: Path to labels file. If None, uses default path.
        """
        assert cpu_use <= 1 and cpu_use > 0
        self.dataset_id: str = dataset_id
        self.cpu_use: float = cpu_use
        self.normalization: Optional[
            Literal["scale", "meanvar-whole", "meanvar-single"]
        ] = normalization

        self.dset_path: str = dset_path or get_processed_dataset_path(self.dataset_id)
        self.labels_path: str = labels_path or get_processed_labels_path(
            self.dataset_id
        )

        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.Preprocessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.shot_list = np.loadtxt(LABELS_PATH)
        self.file_list = self._get_file_list()
        self.num_shots = len(self.file_list)

        self.max_length = self._get_max_length()
        # Only compute dataset-wide statistics if needed for meanvar-whole normalization
        (self.mean, self.std) = (
            self._get_mean_std()
            if self.normalization == "meanvar-whole"
            else (None, None)
        )
        self.sorted_shot_numbers: Optional[NDArray] = None

        # Create lookup dicts for efficiency
        self._shot_to_idx: Dict[int, int] = {
            int(shot[0]): idx for idx, shot in enumerate(self.shot_list)
        }
        self._shot_to_label: Dict[int, NDArray] = {
            int(shot[0]): shot.copy() for shot in self.shot_list
        }

        if not os.path.exists(self.dset_path) or not os.path.exists(self.labels_path):
            self.logger.info(
                f"Preprocessed dataset not found. Creating new dataset "
                f"(dataset_id='{self.dataset_id}', normalization={self.normalization})"
            )
            self._make_dataset(make_labels=True, labels_type="scaled")
        else:
            self.logger.info(
                f"Loading existing preprocessed dataset from {self.dset_path}"
            )
            # Load sorted shot numbers from existing dataset
            self._load_sorted_shot_numbers()

    def _get_file_list(self) -> List[str]:
        """Get list of file names from valid shots in labels file."""
        return [f"{int(shot[0])}.txt" for shot in self.shot_list]

    def _load_sorted_shot_numbers(self) -> None:
        """Load sorted shot numbers from shotlist (dataset is always sorted by shot number)."""
        # Dataset is always sorted by shot number, so just sort shotlist shot numbers
        self.sorted_shot_numbers = np.sort(self.shot_list[:, 0].astype(int))

    def _load_single_file(self, filename: str) -> Tuple[int, NDArray[float32]]:
        """Load and preprocess a single file based on normalization setting."""
        normalization_map = {
            None: lambda: load_and_pad(filename, DATA_DIR, self.max_length),
            "scale": lambda: load_and_pad_scale(filename, DATA_DIR, self.max_length),
            "meanvar-whole": lambda: load_and_pad_norm(
                filename, DATA_DIR, self.max_length, self.mean, self.std
            ),
            "meanvar-single": lambda: load_and_pad_norm(
                filename, DATA_DIR, self.max_length, None, None
            ),
        }
        loader = normalization_map.get(self.normalization)
        if loader is None:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return loader()

    def convert_to_float(self) -> None:
        """Convert dataset and labels tensors to float32."""
        convert_tensors_to_float(self.dset_path, self.labels_path)

    def _process_files_parallel(self, func, *args) -> NDArray:
        """Process files in parallel using ProcessPoolExecutor."""
        use_cores = get_use_cores(self.cpu_use)
        self.logger.info(
            f"Starting parallel processing with {use_cores} worker processes"
        )
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
        self.logger.info(
            f"Computing maximum time series length for {self.num_shots} shots from {DATA_DIR}"
        )
        results = self._process_files_parallel(get_length)
        maximum = int(np.max(results))
        self.logger.info(
            f"Maximum time series length (N_max) determined: {maximum} timesteps"
        )
        return maximum

    def _get_mean_std(self) -> Tuple[float, float]:
        """
        Compute dataset-wide mean and standard deviation.

        Returns:
            Tuple (mean, std). Uses std = sqrt(E[X^2] - E[X]^2).
        """
        self.logger.info(
            f"Computing dataset-wide statistics (mean and std) for {self.num_shots} shots from {DATA_DIR}"
        )
        results = self._process_files_parallel(get_means)
        mean = float(np.mean(results[:, 0]))
        std = float((np.mean(results[:, 1]) - mean**2) ** 0.5)
        self.logger.info(f"Dataset statistics computed: mean={mean:.6f}, std={std:.6f}")
        return (mean, std)

    def make_labels_naive(self, save: bool = False) -> NDArray[np.float64]:
        """
        Create binary classification labels (no time prediction).

        Args:
            save: If True, save labels tensor.

        Returns:
            Labels array (n_shots, 2): [binary_class, original_time].
        """
        labels = create_binary_labels(self.shot_list)
        if save:
            self._save_labels(labels)
        return labels

    def _save_labels(self, labels: NDArray[np.float64]) -> None:
        """Save labels tensor to file."""
        torch.save(torch.tensor(labels), self.labels_path)

    def make_labels_scaled(self, save: bool = False) -> NDArray[np.float64]:
        """
        Create labels with binary classification and scaled disruption time.

        Args:
            save: If True, save labels tensor.

        Returns:
            Labels array (n_shots, 2): [binary_class, scaled_time]. Time in [0,1] or -1.0.
        """
        labels = create_binary_labels(self.shot_list)
        for label_row, shot_row in zip(labels, self.shot_list):
            if shot_row[1] != -1.0:
                label_row[1] = get_scaled_t_disrupt(
                    int(shot_row[0]), DATA_DIR, shot_row[1], self.max_length
                )
        if save:
            self._save_labels(labels)
        return labels

    def _make_dataset(
        self,
        make_labels: bool = True,
        labels_type: Literal["scaled", "naive"] = "scaled",
    ) -> None:
        """
        Build preprocessed dataset from raw signal files.

        Args:
            make_labels: If True, create and save labels tensor.
            labels_type: 'scaled' or 'naive'.
        """
        self.logger.info(
            f"Building preprocessed dataset for {self.num_shots} shots from {DATA_DIR}"
        )
        if self.normalization:
            self.logger.info(f"Normalization method: {self.normalization}")

        estimated_memory_gb = (self.num_shots * self.max_length * 4) / (1024**3)
        self.logger.info(
            f"Estimated memory requirement: ~{estimated_memory_gb:.2f} GB (plus overhead)"
        )

        use_cores = get_use_cores(self.cpu_use)
        self.logger.info(
            f"Using {use_cores} parallel worker processes for data loading"
        )

        # Prepare common arguments
        data_dir_args = [DATA_DIR] * self.num_shots
        max_length_args = [self.max_length] * self.num_shots

        # Map normalization to loader function and arguments
        normalization_loaders = {
            None: (load_and_pad, (self.file_list, data_dir_args, max_length_args)),
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

        loader_func, loader_args = normalization_loaders.get(
            self.normalization, (None, None)
        )
        if loader_func is None:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            results = list[Tuple[int, NDArray[float32]]](
                executor.map(loader_func, *loader_args)
            )

        sorted_data = sorted(results, key=lambda x: x[0])
        sorted_shot_numbers, dataset_data = zip(*sorted_data)
        dataset = np.array(dataset_data)
        self.sorted_shot_numbers = np.array(sorted_shot_numbers)

        dataset_pt = torch.tensor(dataset)

        self.logger.info(f"Saving preprocessed dataset to {self.dset_path}")
        torch.save(dataset_pt, self.dset_path)
        if make_labels:
            self.logger.info(
                f"Creating {labels_type} labels for {len(sorted_shot_numbers)} shots"
            )
            labels_tensor = self._create_labels_in_sorted_order(
                sorted_shot_numbers, labels_type
            )
            self.logger.info(f"Saving labels to {self.labels_path}")
            torch.save(labels_tensor, self.labels_path)
        self.logger.info("Dataset creation completed successfully")

    def _create_labels_in_sorted_order(
        self,
        sorted_shot_numbers: NDArray,
        labels_type: Literal["scaled", "naive"],
    ) -> Tensor:
        """Create labels in sorted shot number order."""
        # Use cached shot_to_label dict
        labels = np.array(
            [self._shot_to_label[int(shot)] for shot in sorted_shot_numbers]
        )
        # Convert to binary labels
        if labels_type == "scaled":
            for label_row, shot_no in zip(labels, sorted_shot_numbers):
                shot_no_int = int(shot_no)
                t_disrupt = label_row[1]
                if t_disrupt != -1.0:
                    label_row[0] = 1
                    label_row[1] = get_scaled_t_disrupt(
                        shot_no_int, DATA_DIR, t_disrupt, self.max_length
                    )
                else:
                    label_row[0] = 0
        else:
            labels[:, 0] = (labels[:, 1] != -1.0).astype(float)
        return torch.tensor(labels)

    def load_example_from_raw(
        self,
        idx: int,
        scale_labels: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load a single example directly from raw files (bypassing preprocessed dataset).

        Args:
            idx: Index in the sorted dataset.
            scale_labels: If True, scale disruption time by self.max_length.

        Returns:
            Tuple of (data_tensor, label_tensor):
                - data_tensor: Preprocessed time series data
                - label_tensor: Corresponding label [classification, time]
        """
        # Always use sorted shot numbers (dataset is always sorted)
        shot_no = int(self.sorted_shot_numbers[idx])
        # Use cached lookup dict for O(1) access
        shotlist_idx = self._shot_to_idx[shot_no]

        # Load label for this example
        t_disrupt = self.shot_list[shotlist_idx, 1]
        is_disruptive = t_disrupt != -1.0
        label = np.array(
            [float(is_disruptive), -1.0 if not is_disruptive else t_disrupt]
        )

        if is_disruptive and scale_labels:
            # Scale disruption time to [0, 1] range
            label[1] = get_scaled_t_disrupt(
                shot_no, DATA_DIR, t_disrupt, self.max_length
            )

        filename = f"{shot_no}.txt"
        data = self._load_single_file(filename)

        return torch.tensor(data[1]), torch.tensor(label)

    def check_dataset(
        self,
        scale_labels: bool = True,
        num_checks: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Verify preprocessed dataset integrity by comparing with raw file processing.

        Args:
            scale_labels: Whether labels were scaled.
            num_checks: Number of random examples to verify.
            verbose: If True, print shot numbers.
        """
        self.logger.info(
            f"Starting dataset integrity check: verifying {num_checks} random examples "
            f"(scale_labels={scale_labels})"
        )
        dataset = IpDataset(self.dset_path, self.labels_path)
        self.logger.debug("Loaded IpDataset for verification")

        total_examples = len(dataset)
        check_indices = random.sample(range(total_examples), num_checks)
        self.logger.debug(
            f"Selected {num_checks} random indices from {total_examples} total examples"
        )

        for idx in check_indices:
            shot_no = int(self.sorted_shot_numbers[idx])
            if verbose:
                self.logger.debug(f"Verifying shot {shot_no} at dataset index {idx}")

            processed_data, processed_label = dataset[idx]
            expected_data, expected_label = self.load_example_from_raw(
                idx, scale_labels
            )

            processed_data_flat = processed_data.squeeze(0)
            processed_label_flat = processed_label.squeeze(0)

            data_match = torch.equal(processed_data_flat, expected_data)
            label_match = torch.equal(processed_label_flat, expected_label)

            if not (data_match and label_match):
                self.logger.warning(
                    f"Dataset mismatch detected at index {idx} (shot {shot_no})"
                )
                if not data_match:
                    max_diff = torch.max(
                        torch.abs(processed_data_flat - expected_data)
                    ).item()
                    self.logger.warning(
                        f"  Data mismatch: maximum difference = {max_diff:.6f}"
                    )
                if not label_match:
                    self.logger.warning("  Label mismatch detected:")
                    self.logger.warning(
                        f"    Processed label: {processed_label_flat.tolist()}"
                    )
                    self.logger.warning(
                        f"    Expected label:  {expected_label.tolist()}"
                    )
                self.logger.error("Dataset integrity check failed: mismatches detected")
                return

        self.logger.info(
            f"Dataset integrity check passed: all {num_checks} verified examples match"
        )
