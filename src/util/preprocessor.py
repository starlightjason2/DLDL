"""
Data preprocessing utilities for plasma disruption datasets.

This module provides the Preprocessor class for loading, normalizing, and
preparing plasma current time series data for training neural networks.
"""

import numpy as np
from numpy.typing import NDArray
import time
import os
import random
import multiprocessing as mp
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from model.model import IpDataset
from util.utils import (
    check_file,
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad,
    load_and_pad_norm,
    load_and_pad_scale,
)

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader
except ImportError:
    pass


################################################################################
## Preprocessor Class
################################################################################
class Preprocessor:
    """
    Preprocessor for plasma current time series data.

    This class handles loading raw plasma current signals, computing dataset
    statistics, normalizing data, creating labels, and building PyTorch-ready
    datasets. It supports multiple normalization strategies and parallel
    processing for efficiency.

    Attributes:
        data_dir: Directory containing raw signal files.
        dataset_path: Path to save/load processed dataset tensor.
        labels_pt_path: Path to save/load processed labels tensor.
        max_length_file: Path to save/load maximum sequence length.
        mean_std_file: Path to save/load dataset mean and std statistics.
        labels_path: Path to raw labels file (shot numbers and disruption times).
    """

    def __init__(
        self, dataset_dir: str, data_dir: str, labels_path: str, dataset_id: str = ""
    ) -> None:
        """
        Initialize the Preprocessor.

        Args:
            dataset_dir: Directory where processed datasets and metadata will be saved.
            data_dir: Directory containing raw signal files (one .txt file per shot).
            labels_path: Path to labels file containing shot numbers and disruption times.
            dataset_id: Optional identifier string appended to output filenames
                (useful for creating multiple preprocessed versions).
        """
        self.data_dir: str = data_dir
        self.dataset_path: str = os.path.join(
            dataset_dir, "processed_dataset" + dataset_id + ".pt"
        )
        self.labels_pt_path: str = os.path.join(
            dataset_dir, "processed_labels" + dataset_id + ".pt"
        )
        self.max_length_file: str = os.path.join(dataset_dir, "max_length.txt")
        self.mean_std_file: str = os.path.join(dataset_dir, "mean_std.txt")
        self.labels_path: str = labels_path

    def convert_to_float(
        self, dataset_path: Optional[str] = None, labels_path: Optional[str] = None
    ) -> None:
        """
        Convert dataset and labels tensors to float32 precision.

        This is useful for reducing memory usage or ensuring consistent dtype
        across different PyTorch versions.

        Args:
            dataset_path: Optional path to dataset file. If None, uses
                self.dataset_path.
            labels_path: Optional path to labels file. If None, uses
                self.labels_pt_path.
        """
        if dataset_path is None:
            dataset = torch.load(self.dataset_path).float()
            torch.save(dataset, self.dataset_path)
        else:
            dataset = torch.load(dataset_path).float()
            torch.save(dataset, dataset_path)

        # Load the labels tensor, convert to float, and re-save
        if labels_path is None:
            labels = torch.load(self.labels_pt_path).float()
            torch.save(labels, self.labels_pt_path)
        else:
            labels = torch.load(labels_path).float()
            torch.save(labels, labels_path)

    def get_max_length(self, save: bool = True, cpu_use: float = 0.8) -> int:
        """
        Compute the maximum time series length across all shots in the dataset.

        This is needed to determine the padding length for creating fixed-size
        tensors. Uses parallel processing for efficiency.

        Args:
            save: If True, save the result to max_length_file for future use.
            cpu_use: Fraction of CPU cores to use for parallel processing (0.0 to 1.0).

        Returns:
            Maximum sequence length found in the dataset.

        Note:
            If max_length_file already exists, loads and returns the cached value
            instead of recomputing.
        """
        if check_file(self.max_length_file):
            return int(np.loadtxt(self.max_length_file).astype(int))

        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num) + ".txt" for num in valid_shots]
        num_shots = len(file_list)
        print(
            "Finding N_max for the {} shots in ".format(int(num_shots)) + self.data_dir
        )
        time_begin = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use) * mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(
                    list(
                        executor.map(get_length, file_list, [self.data_dir] * num_shots)
                    )
                )
            except Exception as e:
                print(f"An error occurred: {e}")

        time_end = time.time()
        elapsed_time = time_end - time_begin

        maximum: int = int(np.max(results))

        print("Finished getting end timesteps in {} seconds.".format(elapsed_time))

        if save:
            np.savetxt(self.max_length_file, np.array([maximum]))

        return maximum

    def get_mean_std(
        self, save: bool = True, cpu_use: float = 0.8
    ) -> NDArray[np.float64]:
        """
        Compute dataset-wide mean and standard deviation for normalization.

        These statistics are computed across all shots and can be used for
        dataset-wide normalization (meanvar-whole). Uses parallel processing
        for efficiency.

        Args:
            save: If True, save the result to mean_std_file for future use.
            cpu_use: Fraction of CPU cores to use for parallel processing (0.0 to 1.0).

        Returns:
            Array containing [mean, std] of the entire dataset.

        Note:
            The standard deviation is computed using the relationship:
            std = sqrt(E[X^2] - E[X]^2) for computational efficiency.
        """
        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num) + ".txt" for num in valid_shots]
        num_shots = len(file_list)
        print(
            "Finding the mean and std. dev. for the {} shots in ".format(int(num_shots))
            + self.data_dir
        )
        time_begin = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use) * mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(
                    list(
                        executor.map(get_means, file_list, [self.data_dir] * num_shots)
                    )
                )
            except Exception as e:
                print(f"An error occurred: {e}")

        time_end = time.time()
        elapsed_time = time_end - time_begin

        mean: float = float(np.mean(results[:, 0]))
        std: float = float((np.mean(results[:, 1]) - mean**2) ** 0.5)

        print("Finished getting stats in {} seconds.".format(elapsed_time))

        if save:
            np.savetxt(self.mean_std_file, np.array([mean, std]))

        return np.array([mean, std])

    def make_labels_naive(self, save: bool = False) -> NDArray[np.float64]:
        """
        Create labels array with binary classification only (no time prediction).

        Converts disruption times to binary labels: 0 for no disruption, 1 for disruption.
        Time information is discarded in this mode.

        Args:
            save: If True, save labels tensor to labels_pt_path.

        Returns:
            Labels array of shape (n_shots, 2) where:
                - labels[:, 0]: Binary classification (0 or 1)
                - labels[:, 1]: Original disruption time (preserved but unused)
        """
        # Load labels file: columns are [shot_number, disruption_time]
        # disruption_time = -1.0 means no disruption occurred
        shotlist = np.loadtxt(self.labels_path)
        labels = np.copy(shotlist)

        # Convert to binary classification labels
        for i in range(shotlist.shape[0]):
            if shotlist[i, 1] == -1.0:
                labels[i, 0] = 0  # No disruption
            else:
                labels[i, 0] = 1  # Disruption occurred

        if save:
            labels_pt = torch.tensor(labels)
            torch.save(labels_pt, self.labels_pt_path)

        return labels

    def make_labels_scaled(
        self, max_length: Optional[int] = None, save: bool = False
    ) -> NDArray[np.float64]:
        """
        Create labels array with binary classification and scaled disruption time.

        The disruption time is normalized by the maximum sequence length to
        produce values in [0, 1], making it suitable for regression tasks.

        Args:
            max_length: Maximum sequence length for scaling. If None, loads from
                max_length_file or computes it.
            save: If True, save labels tensor to labels_pt_path.

        Returns:
            Labels array of shape (n_shots, 2) where:
                - labels[:, 0]: Binary classification (0 or 1)
                - labels[:, 1]: Scaled disruption time index (0.0 to 1.0) for
                  disruptive shots, -1.0 for non-disruptive shots.

        Raises:
            RuntimeError: If max_length is not provided and max_length_file doesn't exist.
        """
        if max_length is None:
            if check_file(self.max_length_file):
                max_length = int(np.loadtxt(self.max_length_file).astype(int))
            else:
                raise RuntimeError(
                    "Max length hasn't been computed yet and " + "wasn't supplied."
                )

        # Load labels file: columns are [shot_number, disruption_time]
        shotlist = np.loadtxt(self.labels_path)
        labels = np.copy(shotlist)

        # Create binary classification + scaled time labels
        for i in range(shotlist.shape[0]):
            if shotlist[i, 1] == -1.0:
                # No disruption: classification = 0, time = -1.0 (invalid)
                labels[i, 0] = 0
            else:
                # Disruption: classification = 1, time = scaled index [0, 1]
                labels[i, 0] = 1
                labels[i, 1] = get_scaled_t_disrupt(
                    int(shotlist[i, 0]), self.data_dir, shotlist[i, 1], max_length
                )

        if save:
            labels_pt = torch.tensor(labels)
            torch.save(labels_pt, self.labels_pt_path)

        return labels

    def make_dataset(
        self,
        normalization: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        max_length: Optional[int] = None,
        make_labels: bool = True,
        labels_type: str = "scaled",
        cpu_use: float = 0.8,
    ) -> None:
        """
        Build the complete preprocessed dataset from raw signal files.

        This is the main preprocessing function that:
        - Loads raw signal files in parallel
        - Applies normalization (if specified)
        - Pads sequences to a fixed length
        - Creates labels (if requested)
        - Saves everything as PyTorch tensors

        Args:
            normalization: Normalization strategy. Options:
                - None: No normalization
                - 'scale': Min-max scaling to [0, 1] per shot
                - 'meanvar-whole': Z-score normalization using dataset-wide statistics
                - 'meanvar-single': Z-score normalization using per-shot statistics
            mean: Dataset-wide mean for normalization (used with 'meanvar-whole').
                If None and normalization='meanvar-whole', computes automatically.
            std: Dataset-wide std for normalization (used with 'meanvar-whole').
                If None and normalization='meanvar-whole', computes automatically.
            max_length: Maximum sequence length for padding. If None, loads from
                max_length_file or computes automatically.
            make_labels: If True, create and save labels tensor.
            labels_type: Type of labels to create. Options:
                - 'scaled': Binary classification + scaled disruption time
                - 'naive': Binary classification only
            cpu_use: Fraction of CPU cores to use for parallel processing (0.0 to 1.0).

        Note:
            The processed dataset is saved to self.dataset_path and labels to
            self.labels_pt_path. This operation can be time-consuming for large datasets.
        """
        if max_length is None:
            if check_file(self.max_length_file):
                max_length = int(np.loadtxt(self.max_length_file).astype(int))
            else:
                max_length = self.get_max_length(cpu_use=cpu_use)

        if normalization == "meanvar-whole":
            if check_file(self.mean_std_file):
                stats = np.loadtxt(self.mean_std_file)
                mean = float(stats[0])
                std = float(stats[1])
            else:
                stats = self.get_mean_std(cpu_use=cpu_use)
                mean = float(stats[0])
                std = float(stats[1])

        # Get list of valid shot files to process
        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num) + ".txt" for num in valid_shots]
        num_shots = len(file_list)
        print(
            "Building dataset for the {} shots in ".format(int(num_shots))
            + self.data_dir
        )
        time_begin = time.time()

        # Set up parallel processing
        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use) * mp.cpu_count()))
        print(f"Running on {use_cores} processes.")

        # Process all files in parallel using the appropriate normalization function
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            try:
                if normalization is None:
                    # No normalization: just pad with zeros
                    results = list(
                        executor.map(
                            load_and_pad,
                            file_list,
                            [self.data_dir] * num_shots,
                            [max_length] * num_shots,
                        )
                    )
                elif normalization == "scale":
                    # Min-max scaling to [0, 1] per shot
                    results = list(
                        executor.map(
                            load_and_pad_scale,
                            file_list,
                            [self.data_dir] * num_shots,
                            [max_length] * num_shots,
                        )
                    )
                elif normalization.startswith("meanvar"):
                    # Z-score normalization (dataset-wide or per-shot)
                    results = list(
                        executor.map(
                            load_and_pad_norm,
                            file_list,
                            [self.data_dir] * num_shots,
                            [max_length] * num_shots,
                            [mean] * num_shots,
                            [std] * num_shots,
                        )
                    )
            except Exception as e:
                print(f"An error occurred: {e}")

        time_end = time.time()
        elapsed_time = time_end - time_begin

        # Create labels if requested
        if make_labels:
            if labels_type == "scaled":
                labels_tensor = torch.tensor(self.make_labels_scaled(max_length))
            else:
                labels_tensor = torch.tensor(self.make_labels_naive())

        # Sort results by shot number and convert to numpy array
        sorted_data = sorted(results, key=lambda x: x[0])
        dataset = np.zeros((num_shots, max_length))
        for i in range(num_shots):
            dataset[i, :] = sorted_data[i][1]

        # Convert to PyTorch tensor and save
        dataset_pt = torch.tensor(dataset)

        print("Finished loading and preparing data in {} seconds.".format(elapsed_time))

        torch.save(dataset_pt, self.dataset_path)
        if make_labels:
            torch.save(labels_tensor, self.labels_pt_path)

    def load_example_from_raw(
        self,
        idx: int,
        normalization: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        scale_labels: bool = True,
        max_length: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load a single example directly from raw files (bypassing preprocessed dataset).

        This is useful for debugging and verifying that preprocessing produces
        the expected results. The processing matches what make_dataset() does.

        Args:
            idx: Index of the example to load (corresponds to row in labels file).
            normalization: Normalization strategy (see make_dataset for options).
            mean: Dataset-wide mean for normalization.
            std: Dataset-wide std for normalization.
            scale_labels: If True, scale disruption time by max_length.
            max_length: Maximum sequence length for padding. If None, loads from file.

        Returns:
            Tuple of (data_tensor, label_tensor):
                - data_tensor: Preprocessed time series data
                - label_tensor: Corresponding label [classification, time]

        Raises:
            RuntimeError: If max_length is not available and normalization requires
                dataset-wide statistics that haven't been computed.
        """
        if max_length is None:
            if check_file(self.max_length_file):
                max_length = int(np.loadtxt(self.max_length_file).astype(int))
            else:
                raise RuntimeError(
                    "Max length hasn't been computed yet and " + "wasn't supplied."
                )

        # Load label for this example
        shotlist = np.loadtxt(self.labels_path)
        label = np.array([0, 0.0])
        if shotlist[idx, 1] == -1.0:
            # No disruption
            label[0] = 0
            label[1] = -1.0
        else:
            # Disruption occurred
            label[0] = 1
            if scale_labels:
                # Scale disruption time to [0, 1] range
                label[1] = get_scaled_t_disrupt(
                    int(shotlist[idx, 0]), self.data_dir, shotlist[idx, 1], max_length
                )
            else:
                # Use raw disruption time
                label[1] = shotlist[idx, 1]

        if normalization == "meanvar-whole" and mean is None:
            if check_file(self.mean_std_file):
                stats = np.loadtxt(self.mean_std_file)
                mean = float(stats[0])
                std = float(stats[1])
            else:
                raise RuntimeError(
                    "Statistics haven't been computed yet and " + "weren't supplied."
                )

        shot_no = int(shotlist[idx, 0])
        filename = str(shot_no) + ".txt"
        if normalization is None:
            data = load_and_pad(filename, self.data_dir, max_length)
        elif normalization == "scale":
            data = load_and_pad_scale(filename, self.data_dir, max_length)
        elif normalization.startswith("meanvar"):
            data = load_and_pad_norm(filename, self.data_dir, max_length, mean, std)

        return torch.tensor(data[1]), torch.tensor(label)

    def check_dataset(
        self,
        dset_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        num_checks: int = 100,
        normalization: Optional[str] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        scale_labels: bool = True,
        max_length: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Verify the integrity of a preprocessed dataset.

        This method validates that the preprocessed dataset matches what would be
        produced by processing raw files. It randomly samples examples and compares
        them against load_example_from_raw() output.

        Args:
            dset_path: Path to preprocessed dataset file. If None, uses
                self.dataset_path.
            labels_path: Path to preprocessed labels file. If None, uses
                self.labels_pt_path.
            num_checks: Number of random examples to verify.
            normalization: Normalization strategy used during preprocessing.
            mean: Dataset-wide mean used during preprocessing.
            std: Dataset-wide std used during preprocessing.
            scale_labels: Whether labels were scaled during preprocessing.
            max_length: Maximum sequence length used during preprocessing.
            verbose: If True, print shot numbers being checked.

        Note:
            This is a validation/debugging tool. A mismatch indicates the
            preprocessing pipeline may have inconsistencies.
        """
        print("Checking dataset alignment...")

        # Determine file paths
        if dset_path is None:
            dataset_file_path = self.dataset_path
        else:
            dataset_file_path = dset_path
        if labels_path is None:
            labels_file_path = self.labels_pt_path
        else:
            labels_file_path = labels_path

        # Load the preprocessed dataset
        dataset = IpDataset(dataset_file_path, labels_file_path)
        print("loaded IpDataset")

        # Randomly sample indices to check (for efficiency)
        if verbose:
            shotlist = np.loadtxt(self.labels_path)
        total_examples = len(dataset)
        check_indices = random.sample(range(total_examples), num_checks)

        # Verify each sampled example matches raw processing
        dataset_correct = True
        for idx in check_indices:
            if verbose:
                print(f"Checking shot {int(shotlist[idx,0])}.")

            # Get example from preprocessed dataset
            processed_data, processed_label = dataset[idx]

            # Recompute from raw files using the same preprocessing pipeline
            expected_data, expected_label = self.load_example_from_raw(
                idx, normalization, mean, std, scale_labels, max_length
            )

            # Compare: they should be identical
            if not torch.equal(
                processed_data.squeeze(0), expected_data
            ) or not torch.equal(processed_label.squeeze(0), expected_label):
                print(f"Mismatch found at index {idx}")
                dataset_correct = False
                break

        if dataset_correct:
            print(
                "Dataset check passed: Processed data matches expected data"
                + " for checked examples."
            )
        else:
            print(
                "Dataset check failed: Some processed examples do not match"
                + " the expected outputs."
            )
