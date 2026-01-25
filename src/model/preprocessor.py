"""Data preprocessing utilities for plasma disruption datasets."""

from numpy import float32
import numpy as np
from numpy.typing import NDArray
import time
import random
import os
from typing import List, Literal, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
from model.model import IpDataset
from util.data_loading import (
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad,
    load_and_pad_norm,
    load_and_pad_scale,
)
from util.preprocessing import (
    get_use_cores,
    create_binary_labels,
    convert_tensors_to_float,
)
from constants import (
    DATA_DIR,
    LABELS_PATH,
    get_processed_dataset_path,
    get_processed_labels_path,
)

try:
    import torch
    from torch import Tensor
except ImportError:
    pass


class Preprocessor:
    """Preprocessor for plasma current time series data."""

    def __init__(
        self,
        dataset_id: str = "",
        cpu_use: float = 0.8,
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

        self.shot_list = np.loadtxt(LABELS_PATH)
        self.file_list = self._get_file_list()
        self.num_shots = len(self.file_list)

        self.max_length = self._get_max_length()
        (self.mean, self.std) = self._get_mean_std()
        self.sorted_shot_numbers: Optional[NDArray] = None

        # Create lookup dicts for efficiency
        self._shot_to_idx: Dict[int, int] = {
            int(self.shot_list[i, 0]): i for i in range(len(self.shot_list))
        }
        self._shot_to_label: Dict[int, NDArray] = {
            int(self.shot_list[i, 0]): self.shot_list[i].copy()
            for i in range(len(self.shot_list))
        }

        if not os.path.exists(self.dset_path) or not os.path.exists(self.labels_path):
            self._make_dataset(make_labels=True, labels_type="scaled")
        else:
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
        if self.normalization is None:
            return load_and_pad(filename, DATA_DIR, self.max_length)
        elif self.normalization == "scale":
            return load_and_pad_scale(filename, DATA_DIR, self.max_length)
        elif self.normalization.startswith("meanvar"):
            return load_and_pad_norm(
                filename, DATA_DIR, self.max_length, self.mean, self.std
            )
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def convert_to_float(self) -> None:
        """Convert dataset and labels tensors to float32."""
        convert_tensors_to_float(self.dset_path, self.labels_path)

    def _process_files_parallel(self, func, *args) -> NDArray:
        """Process files in parallel using ProcessPoolExecutor."""
        use_cores = get_use_cores(self.cpu_use)
        print(f"Running on {use_cores} processes.")
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
        print(f"Finding N_max for {self.num_shots} shots in {DATA_DIR}")
        results = self._process_files_parallel(get_length)
        maximum = int(np.max(results))
        print(f"N_max={maximum}")
        return maximum

    def _get_mean_std(self) -> Tuple[float, float]:
        """
        Compute dataset-wide mean and standard deviation.

        Returns:
            Tuple (mean, std). Uses std = sqrt(E[X^2] - E[X]^2).
        """
        print(
            f"Finding the mean and std. dev. for {self.num_shots} shots in {DATA_DIR}"
        )
        results = self._process_files_parallel(get_means)
        mean = float(np.mean(results[:, 0]))
        std = float((np.mean(results[:, 1]) - mean**2) ** 0.5)
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
        for i in range(self.shot_list.shape[0]):
            if self.shot_list[i, 1] != -1.0:
                labels[i, 1] = get_scaled_t_disrupt(
                    int(self.shot_list[i, 0]),
                    DATA_DIR,
                    self.shot_list[i, 1],
                    self.max_length,
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
        print(f"Building dataset for {self.num_shots} shots in {DATA_DIR}")

        estimated_memory_gb = (self.num_shots * self.max_length * 4) / (1024**3)
        print(f"Estimated memory: ~{estimated_memory_gb:.2f} GB (plus overhead)")

        use_cores = get_use_cores(self.cpu_use)
        print(f"Running on {use_cores} processes.")

        # Prepare common arguments
        data_dir_args = [DATA_DIR] * self.num_shots
        max_length_args = [self.max_length] * self.num_shots

        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            if self.normalization is None:
                results = list[Tuple[int, NDArray[float32]]](
                    executor.map(
                        load_and_pad,
                        self.file_list,
                        data_dir_args,
                        max_length_args,
                    )
                )
            elif self.normalization == "scale":
                results = list[Tuple[int, NDArray[float32]]](
                    executor.map(
                        load_and_pad_scale,
                        self.file_list,
                        data_dir_args,
                        max_length_args,
                    )
                )
            elif self.normalization.startswith("meanvar"):
                results = list[Tuple[int, NDArray[float32]]](
                    executor.map(
                        load_and_pad_norm,
                        self.file_list,
                        data_dir_args,
                        max_length_args,
                        [self.mean] * self.num_shots,
                        [self.std] * self.num_shots,
                    )
                )

        sorted_data = sorted(results, key=lambda x: x[0])
        dataset = np.array([item[1] for item in sorted_data])
        # Store sorted shot numbers for alignment with saved dataset
        sorted_shot_numbers = np.array([item[0] for item in sorted_data])
        self.sorted_shot_numbers = sorted_shot_numbers

        dataset_pt = torch.tensor(dataset)

        torch.save(dataset_pt, self.dset_path)
        if make_labels:
            labels_tensor = self._create_labels_in_sorted_order(
                sorted_shot_numbers, labels_type
            )
            torch.save(labels_tensor, self.labels_path)

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
            for i in range(len(labels)):
                shot_no = int(labels[i, 0])
                t_disrupt = labels[i, 1]
                if t_disrupt != -1.0:
                    labels[i, 0] = 1
                    labels[i, 1] = get_scaled_t_disrupt(
                        shot_no, DATA_DIR, t_disrupt, self.max_length
                    )
                else:
                    labels[i, 0] = 0
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
        label = np.array([0, 0.0])
        if self.shot_list[shotlist_idx, 1] == -1.0:
            # No disruption
            label[0] = 0
            label[1] = -1.0
        else:
            # Disruption occurred
            label[0] = 1
            if scale_labels:
                # Scale disruption time to [0, 1] range
                label[1] = get_scaled_t_disrupt(
                    shot_no,
                    DATA_DIR,
                    self.shot_list[shotlist_idx, 1],
                    self.max_length,
                )
            else:
                # Use raw disruption time
                label[1] = self.shot_list[shotlist_idx, 1]

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
        print("Checking dataset alignment...")
        dataset = IpDataset(self.dset_path, self.labels_path)
        print("loaded IpDataset")

        total_examples = len(dataset)
        check_indices = random.sample(range(total_examples), num_checks)

        dataset_correct = True
        for idx in check_indices:
            if verbose:
                shot_no = int(self.sorted_shot_numbers[idx])
                print(f"Checking shot {shot_no}.")

            processed_data, processed_label = dataset[idx]
            expected_data, expected_label = self.load_example_from_raw(
                idx, scale_labels
            )

            if not torch.equal(
                processed_data.squeeze(0), expected_data
            ) or not torch.equal(processed_label.squeeze(0), expected_label):
                print(f"Mismatch found at index {idx}")
                dataset_correct = False
                break

        if dataset_correct:
            print("Dataset check passed.")
        else:
            print("Dataset check failed.")
