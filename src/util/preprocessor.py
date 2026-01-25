"""Data preprocessing utilities for plasma disruption datasets."""

from numpy import float32
import numpy as np
from numpy.typing import NDArray
import time
import random
import multiprocessing as mp
from typing import List, Literal, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from model.model import IpDataset
from util.utils import (
    get_length,
    get_scaled_t_disrupt,
    get_means,
    load_and_pad,
    load_and_pad_norm,
    load_and_pad_scale,
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


################################################################################
## Preprocessor Class
################################################################################
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
        self.shotlist = np.loadtxt(LABELS_PATH)
        self.max_length = self.get_max_length()
        self.get_mean_std()  # Compute and cache mean/std on initialization

    def _get_file_list(self) -> List[str]:
        """Get list of file names from valid shots in labels file."""
        return [f"{int(shot[0])}.txt" for shot in self.shotlist]

    def _get_use_cores(self) -> int:
        """Calculate number of CPU cores to use for parallel processing."""
        return max(1, int(self.cpu_use * mp.cpu_count()))

    def convert_to_float(self) -> None:
        """Convert dataset and labels tensors to float32."""
        torch.save(torch.load(self.dset_path).float(), self.dset_path)
        torch.save(torch.load(self.labels_path).float(), self.labels_path)

    def _process_files_parallel(self, func, file_list: List[str], *args) -> NDArray:
        """Process files in parallel using ProcessPoolExecutor."""
        num_shots = len(file_list)
        use_cores = self._get_use_cores()
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            try:
                return np.asarray(
                    list(executor.map(func, file_list, [DATA_DIR] * num_shots, *args))
                )
            except Exception as e:
                print(f"An error occurred: {e}")
                raise

    def get_max_length(self) -> int:
        """Compute maximum time series length across all shots."""
        file_list = self._get_file_list()
        num_shots = len(file_list)
        print(f"Finding N_max for {num_shots} shots in {DATA_DIR}")
        time_begin = time.time()

        results = self._process_files_parallel(get_length, file_list)
        maximum = int(np.max(results))

        elapsed_time = time.time() - time_begin
        print(
            f"Finished getting end timesteps in {elapsed_time:.2f} seconds. N_max={maximum}"
        )
        return maximum

    def get_mean_std(self) -> NDArray[np.float64]:
        """
        Compute dataset-wide mean and standard deviation.

        Returns:
            Array [mean, std]. Uses std = sqrt(E[X^2] - E[X]^2).
        """
        file_list = self._get_file_list()
        num_shots = len(file_list)
        print(f"Finding the mean and std. dev. for {num_shots} shots in {DATA_DIR}")
        time_begin = time.time()

        results = self._process_files_parallel(get_means, file_list)
        self.mean = float(np.mean(results[:, 0]))
        self.std = float((np.mean(results[:, 1]) - self.mean**2) ** 0.5)

        elapsed_time = time.time() - time_begin
        print(f"Finished getting stats in {elapsed_time:.2f} seconds.")
        return np.array([self.mean, self.std])

    def _create_binary_labels(self) -> NDArray[np.float64]:
        """Create binary classification labels from shotlist."""
        labels = np.copy(self.shotlist)
        labels[:, 0] = (self.shotlist[:, 1] != -1.0).astype(float)
        return labels

    def make_labels_naive(self, save: bool = False) -> NDArray[np.float64]:
        """
        Create binary classification labels (no time prediction).

        Args:
            save: If True, save labels tensor.

        Returns:
            Labels array (n_shots, 2): [binary_class, original_time].
        """
        labels = self._create_binary_labels()
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
        labels = self._create_binary_labels()
        for i in range(self.shotlist.shape[0]):
            if self.shotlist[i, 1] != -1.0:
                labels[i, 1] = get_scaled_t_disrupt(
                    int(self.shotlist[i, 0]),
                    DATA_DIR,
                    self.shotlist[i, 1],
                    self.max_length,
                )
        if save:
            self._save_labels(labels)
        return labels

    def make_dataset(
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
        file_list = self._get_file_list()
        num_shots = len(file_list)
        print(f"Building dataset for {num_shots} shots in {DATA_DIR}")

        estimated_memory_gb = (num_shots * self.max_length * 4) / (1024**3)
        print(f"Estimated memory: ~{estimated_memory_gb:.2f} GB (plus overhead)")

        time_begin = time.time()
        use_cores = self._get_use_cores()
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers=use_cores) as executor:
            try:
                if self.normalization is None:
                    results = list[Tuple[int, NDArray[float32]]](
                        executor.map(
                            load_and_pad,
                            file_list,
                            [DATA_DIR] * num_shots,
                            [self.max_length] * num_shots,
                        )
                    )
                elif self.normalization == "scale":
                    results = list[Tuple[int, NDArray[float32]]](
                        executor.map(
                            load_and_pad_scale,
                            file_list,
                            [DATA_DIR] * num_shots,
                            [self.max_length] * num_shots,
                        )
                    )
                elif self.normalization.startswith("meanvar"):
                    results = list[Tuple[int, NDArray[float32]]](
                        executor.map(
                            load_and_pad_norm,
                            file_list,
                            [DATA_DIR] * num_shots,
                            [self.max_length] * num_shots,
                            [self.mean] * num_shots,
                            [self.std] * num_shots,
                        )
                    )
            except Exception as e:
                print(f"An error occurred: {e}")

        time_end = time.time()
        elapsed_time = time_end - time_begin

        if make_labels:
            labels_tensor = torch.tensor(
                self.make_labels_scaled()
                if labels_type == "scaled"
                else self.make_labels_naive()
            )

        sorted_data = sorted(results, key=lambda x: x[0])
        dataset = np.array([item[1] for item in sorted_data])

        dataset_pt = torch.tensor(dataset)

        print("Finished loading and preparing data in {} seconds.".format(elapsed_time))

        torch.save(dataset_pt, self.dset_path)
        if make_labels:
            torch.save(labels_tensor, self.labels_path)

    def load_example_from_raw(
        self,
        idx: int,
        scale_labels: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load a single example directly from raw files (bypassing preprocessed dataset).

        Args:
            idx: Index of the example to load (corresponds to row in labels file).
            scale_labels: If True, scale disruption time by self.max_length.

        Returns:
            Tuple of (data_tensor, label_tensor):
                - data_tensor: Preprocessed time series data
                - label_tensor: Corresponding label [classification, time]
        """
        # Load label for this example
        label = np.array([0, 0.0])
        if self.shotlist[idx, 1] == -1.0:
            # No disruption
            label[0] = 0
            label[1] = -1.0
        else:
            # Disruption occurred
            label[0] = 1
            if scale_labels:
                # Scale disruption time to [0, 1] range
                label[1] = get_scaled_t_disrupt(
                    int(self.shotlist[idx, 0]),
                    DATA_DIR,
                    self.shotlist[idx, 1],
                    self.max_length,
                )
            else:
                # Use raw disruption time
                label[1] = self.shotlist[idx, 1]

        filename = f"{int(self.shotlist[idx, 0])}.txt"
        if self.normalization is None:
            data = load_and_pad(filename, DATA_DIR, self.max_length)
        elif self.normalization == "scale":
            data = load_and_pad_scale(filename, DATA_DIR, self.max_length)
        elif self.normalization.startswith("meanvar"):
            data = load_and_pad_norm(
                filename, DATA_DIR, self.max_length, self.mean, self.std
            )

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
                print(f"Checking shot {int(self.shotlist[idx,0])}.")

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
