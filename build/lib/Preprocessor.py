import numpy as np
import time
import os
import random
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from DLDL import ipDataset
try:
    import torch
    from torch.utils.data import DataLoader
except:
    pass

################################################################################
## Utility Functions and Globals
################################################################################
def check_file(file_path, verbose = False):
    if os.path.exists(file_path):
        if verbose:
            file_size = os.path.getsize(file_path)
            print(f"File {file_path} exists. Size: {file_size} bytes.")
        return True
    else:
        if verbose:
            print(f"File {file_path} does not exist.")
        return False


def get_length(filename, data_dir):
    """
    Get length of time series for a single file.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
    """
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, usecols=1)

    return len(data)


def get_scaled_t_disrupt(shot_no, data_dir, t_disrupt, max_length):
    """
    Get scaled version of t_disrupt; i_disrupt/max_length
    """
    shot_file = os.path.join(data_dir,str(shot_no)+'.txt')
    time = np.loadtxt(shot_file, usecols=0)
    i_d = np.abs(time-t_disrupt).argmin()

    return i_d/max_length


def get_means(filename, data_dir):
    """
    Get mean and mean of squares of time series for a single file.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
    """
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, usecols=1)

    mean = np.mean(data)
    mean2 = np.mean(data**2)

    return [mean, mean2]


def load_and_pad(filename, data_dir, max_length):
    """
    Loads a single current series and pads it with zeros up to the max length,
    then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, usecols=1, dtype=np.float32)
    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded_data)


def load_and_pad_norm(filename, data_dir, max_length, mean = None, std = None):
    """
    Loads a single current series and pads it with zeros up to the max length,
    normalizes the signal values then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
        mean: float, supply if you want to use dataset-wide statistics
        std: float, " "
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, usecols=1, dtype=np.float32)
    
    if mean == None:
        mean = np.mean(data)
        std = np.std(data)

    data = (data - mean)/std

    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded_data)


def load_and_pad_scale(filename, data_dir, max_length):
    """
    Loads a single current series and pads it with zeros up to the max length,
    scales data values to [0,1], then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, usecols=1, dtype=np.float32)
    data = data - np.min(data)
    data = data/np.max(data)
    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded_data)


################################################################################
## Preprocessor Class
################################################################################
class Preprocessor:
    def __init__(self, dataset_dir, data_dir, labels_path, ID = ""):
        self.data_dir = data_dir
        self.dataset_path = os.path.join(dataset_dir,'processed_dataset'+ID+'.pt')
        self.labels_pt_path = os.path.join(dataset_dir,'processed_labels'+ID+'.pt')
        self.max_length_file = os.path.join(dataset_dir,'max_length.txt')
        self.mean_std_file = os.path.join(dataset_dir,'mean_std.txt')
        self.labels_path = labels_path


    def Convert_2_float(self, dataset_path = None, labels_path = None):
        # Load the dataset tensor, convert to float, and re-save
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


    def Get_Max_Length(self, save = True, cpu_use = 0.8):
        """
        Acquires the maximum length of the current time series across the
        entire dataset.

        Args:
            save: bool, True to save result in data_dir
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        if check_file(self.max_length_file):
            return np.loadtxt(self.max_length_file).astype(int)

        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num)+'.txt' for num in valid_shots]
        num_shots = len(file_list)
        print("Finding N_max for the {} shots in ".format(int(num_shots))\
                +self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(list(executor.map(get_length, file_list,\
                           [self.data_dir]*num_shots)))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        maximum = np.max(results)

        print("Finished getting end timesteps in {} seconds.".format(T))

        if save:
            np.savetxt(self.max_length_file, np.array([maximum]))

        return maximum

        
    def Get_Mean_Std(self, save = True, cpu_use = 0.8):
        """
        Acquires the mean and std. dev. of the entire dataset.

        Args:
            save: bool, True to save result in data_dir
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num)+'.txt' for num in valid_shots]
        num_shots = len(file_list)
        print("Finding the mean and std. dev. for the {} shots in ".format(\
                int(num_shots))+self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(list(executor.map(get_means, file_list,\
                           [self.data_dir]*num_shots)))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        mean = np.mean(results[:,0])
        std = (np.mean(results[:,1])-mean**2)**0.5

        print("Finished getting stats in {} seconds.".format(T))

        if save:
            np.savetxt(self.mean_std_file, np.array([mean, std]))

        return np.array([mean, std])


    def Make_Labels_Naive(self, save = False):
        """
        Makes labels tensor using a naive t_disrupt label
        """
        shotlist = np.loadtxt(self.labels_path)
        labels = np.copy(shotlist)

        for i in range(shotlist.shape[0]):
            if shotlist[i,1] == -1.0:
                labels[i,0] = 0
            else:
                labels[i,0] = 1

        if save:
            labels_pt = torch.tensor(labels)
            torch.save(labels_pt, self.labels_pt_path)

        return labels

        
    def Make_Labels_Scaled(self, max_length = None, save = False):
        """
        Makes labels tensor using a scaled t_disrupt label
        """
        if max_length == None:
            if check_file(self.max_length_file):
                max_length = np.loadtxt(self.max_length_file).astype(int)
            else:
                raise RuntimeError("Max length hasn't been computed yet and "+\
                        "wasn't supplied.")

        shotlist = np.loadtxt(self.labels_path)
        labels = np.copy(shotlist)

        for i in range(shotlist.shape[0]):
            if shotlist[i,1] == -1.0:
                labels[i,0] = 0
            else:
                labels[i,0] = 1
                labels[i,1] = get_scaled_t_disrupt(int(shotlist[i,0]),\
                        self.data_dir, shotlist[i,1], max_length)

        if save:
            labels_pt = torch.tensor(labels)
            torch.save(labels_pt, self.labels_pt_path)

        return labels

        
    def Make_Dataset(self, normalization = None, mean = None, std = None,\
                     max_length = None, make_labels = True, labels = 'scaled',\
                     cpu_use = 0.8):
        """
        Acquires the maximum length of the current time series across the
        entire dataset.

        Args:
            normalization: str, specifies normalization type, options are
                           'scale' 'meanvar-whole' and 'meanvar-single', leave 
                           blank for no normalization
            mean, std: float, dataset-wide statistics if desired
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        if max_length == None:
            if check_file(self.max_length_file):
                max_length = np.loadtxt(self.max_length_file).astype(int)
            else:
                max_length = self.Get_Max_Length(cpu_use = cpu_use)

        if normalization == "meanvar-whole":
            if check_file(self.mean_std_file):
                stats = np.loadtxt(self.mean_std_file)
                mean = stats[0]
                std = stats[1]
            else:
                stats = self.Get_Mean_Std(cpu_use = cpu_use)
                mean = stats[0]
                std = stats[1]

        valid_shots = np.loadtxt(self.labels_path, usecols=0).astype(int)
        file_list = [str(num)+'.txt' for num in valid_shots]
        num_shots = len(file_list)
        print("Building dataset for the {} shots in ".format(int(num_shots))\
                +self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                if normalization == None:
                    results = list(executor.map(load_and_pad,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots))
                elif normalization == "scale":
                    results = list(executor.map(load_and_pad_scale,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots))
                elif normalization.startswith("meanvar"):
                    results = list(executor.map(load_and_pad_norm,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots, [mean]*num_shots,\
                            [std]*num_shots))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        if make_labels:
            if labels == 'scaled':
                labels = torch.tensor(self.Make_Labels_Scaled(max_length))
            else:
                labels = torch.tensor(self.Make_Labels_Naive())

        sorted_data = sorted(results, key=lambda x: x[0])
        dataset = np.zeros((num_shots, max_length))
        for i in range(num_shots):
            dataset[i,:] = sorted_data[i][1]

        dataset_pt = torch.tensor(dataset)

        print("Finished loading and preparing data in {} seconds.".format(T))

        torch.save(dataset_pt, self.dataset_path)
        if make_labels:
            torch.save(labels, self.labels_pt_path)


    def load_example_from_raw(self, idx, normalization = None, mean = None,\
            std = None, scale_labels = True, max_length = None):
        """
        Loads a single example from the raw files directly for comparison with
        preprocessed data

        Args are described elsewhere.
        """
        if max_length == None:
            if check_file(self.max_length_file):
                max_length = np.loadtxt(self.max_length_file).astype(int)
            else:
                raise RuntimeError("Max length hasn't been computed yet and "+\
                        "wasn't supplied.")

        shotlist = np.loadtxt(self.labels_path)
        label = np.array([0,0.0])
        if shotlist[idx,1] == -1.0:
            label[0] = 0
            label[1] = -1.0
        else:
            label[0] = 1
            if scale_labels:
                label[1] = get_scaled_t_disrupt(int(shotlist[idx,0]),\
                            self.data_dir, shotlist[idx,1], max_length)
            else:
                label[1] = shotlist[idx,1]

        if normalization == "meanvar-whole" and mean == None:
            if check_file(self.mean_std_file):
                stats = np.loadtxt(self.mean_std_file)
                mean = stats[0]
                std = stats[1]
            else:
                raise RuntimeError("Statistics haven't been computed yet and "+\
                        "weren't supplied.")

        shot_no = int(shotlist[idx,0])
        filename = str(shot_no)+'.txt'
        if normalization == None:
            data = load_and_pad(filename, self.data_dir, max_length)
        elif normalization == "scale":
            data = load_and_pad_scale(filename, self.data_dir, max_length)
        elif normalization.startswith("meanvar"):
            data = load_and_pad_norm(filename, self.data_dir, max_length,\
                    mean, std)

        return torch.tensor(data[1]), torch.tensor(label)


    def Check_Dataset(self, dset_path = None, labels_path = None, num_checks=100,\
            normalization = None, mean = None, std = None, scale_labels = True,\
            max_length = None, verbose = False):
        """
        Checks the integrity of the processed dataset by comparing randomly selected examples
        against the output of 'load_example_from_raw'.

        Parameters:
        - load_example_from_raw: A function that takes an identifier or file path as input and
          returns a processed example and its label.
        - num_checks: The number of random examples to check for validation.
        """
        print("Checking dataset alignment...")
        # Load Dataset
        if dset_path is None:
            d_path = self.dataset_path
        else:
            d_path = dset_path
        if labels_path is None:
            l_path = self.labels_pt_path
        else:
            l_path = labels_path
        dset = ipDataset(d_path, l_path)
        print("loaded ipDataset")

        # Generate a list of random indices to check
        if verbose:
            shotlist = np.loadtxt(self.labels_path)
        total_examples = len(dset)
        check_indices = random.sample(range(total_examples), num_checks)

        # Flag to indicate if the dataset is correctly processed
        dataset_correct = True

        for idx in check_indices:
            if verbose:
                print(f"Checking shot {int(shotlist[idx,0])}.")
            # Load processed example and label from the DataLoader
            processed_data, processed_label = dset[idx]

            # Load the expected example and label using the raw data function
            expected_data, expected_label = self.load_example_from_raw(idx,\
                    normalization, mean, std, scale_labels, max_length)

            # Compare the processed and expected examples
            if not torch.equal(processed_data.squeeze(0), expected_data) or\
                    not torch.equal(processed_label.squeeze(0), expected_label):
                print(f"Mismatch found at index {idx}")
                dataset_correct = False
                break

        if dataset_correct:
            print("Dataset check passed: Processed data matches expected data"+\
                  " for checked examples.")
        else:
            print("Dataset check failed: Some processed examples do not match"+\
                  " the expected outputs.")