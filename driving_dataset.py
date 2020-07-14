""" Definition of pytorch dataset forestimation of accident risk from driving data """
import os
import re

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np


class DrivingDataset(Dataset):
    """ Driving Behavior Classification Dataset """
    def __init__(self, path, folds, selected_parameters=None, in_memory=True, acc_types=None,
                 include_index=False, normalize_like=None):
        self.dataset = ConcatDataset([DrivingDatasetOneFold(path, fold, selected_parameters,
                                                            in_memory, acc_types, include_index)
                                      for fold in folds])

        if normalize_like is None:
            total_n_windows = sum(fold.n_windows for fold in self.dataset.datasets)
            avg_x = sum(f.avg_x*f.n_windows for f in self.dataset.datasets) / total_n_windows
            avg_x2 = sum(f.avg_x2*f.n_windows for f in self.dataset.datasets) / total_n_windows

            self.mean = avg_x.transpose(0, 2, 1).astype(np.float32)
            self.std = np.sqrt(avg_x2 - avg_x**2).transpose(0, 2, 1).astype(np.float32)
            # Convert NumPy arrays to Torch tensors to take advantage of torch multiprocessing optimizations
            self.mean = torch.from_numpy(self.mean)
            self.std = torch.from_numpy(self.std)
        else:
            self.mean = normalize_like.mean
            self.std = normalize_like.std
        # Free up no longer needed underlying DrivingDatasetOneFold avg_x and avg_x2
        for fold in self.dataset.datasets:
            fold.avg_x = None
            fold.avg_x2 = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, *other = self.dataset[index]
        x = (x - self.mean) / self.std
        return (x, *other)


class DrivingDatasetOneFold(Dataset):
    """ One fold of Driving Behavior Classification Dataset """
    def __init__(self, path, fold, selected_parameters=None, in_memory=True, acc_types=None,
                 include_index=False):
        if acc_types is None:
            raise ValueError('Accident types used for labels unspecified')
        if isinstance(selected_parameters, str):
            self.selected_parameters = [int(x) for x in selected_parameters.split()]
        elif isinstance(selected_parameters, list):
            self.selected_parameters = selected_parameters
        elif selected_parameters is None:
            self.selected_parameters = None
        else:
            raise ValueError('Selected parameters argument needs to be a list or a string.')
        self.include_index = include_index
        self.seq_length = self.__extract_seq_length(path)

        # Data Loading
        self.examples = self.__read_examples(path, fold, acc_types)
        self.windows_data = np.load(os.path.join(path, f'X_{fold}.npy'),
                                    mmap_mode=(None if in_memory else 'r'))
        self.n_windows = len(self.windows_data)

        avg_x_path = os.path.join(path, f'X_avg_x_{fold}.npy')
        self.avg_x = np.load(avg_x_path)[:, :, self.selected_parameters]
        avg_x2_path = os.path.join(path, f'X_avg_x2_{fold}.npy')
        self.avg_x2 = np.load(avg_x2_path)[:, :, self.selected_parameters]

        # Convert NumPy arrays to Torch tensors to use torch multiprocessing optimizations
        self.examples = torch.from_numpy(self.examples)
        if in_memory: # Keep numpy array if memmapped
            self.windows_data = torch.from_numpy(self.windows_data)
        # Keep avg_x and avg_x2 are used and free'd by DrivingDataset constructor


    @staticmethod
    def __read_examples(path, fold, acc_types):
        label_cols = tuple(i+1 for i in acc_types)
        examples = np.load(os.path.join(path, f'seq_{fold}.npy'))
        indexes = examples[:, 0]
        labels = np.any(examples[:, label_cols] < 365, axis=1)
        return np.stack((indexes, labels), axis=1)

    @staticmethod
    def __extract_seq_length(path):
        match = re.search(r"_([0-9]+)w_", os.path.basename(path))
        if match is None:
            raise ValueError('Dataset directory name does not contain sequence length (_[0-9]+w_)')
        return int(match.group(1))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        start_index, target = self.examples[index]
        windows_indexes = torch.arange(start_index, start_index + self.seq_length)
        window_seq = self.windows_data[windows_indexes] # (n_windows, window_length, channels)
        if type(window_seq) == np.ndarray: # if memmapping is used, self.windows_data will be a Numpy array
            window_seq = torch.from_numpy(window_seq)
        window_seq = window_seq.transpose(1, 2) # (n_windows, channels, window_length) for torch

        target = target.reshape(-1).float() # Loss need labels as float

        if self.selected_parameters is not None:
            window_seq = window_seq[:, self.selected_parameters, :]
        if self.include_index:
            return window_seq, target, index

        return window_seq, target


def main():
    """ Demonstrate data loading performances """
    from tqdm import tqdm
    dataset = DrivingDataset('/media/raid/18m_road_seq_20w_any_acc_type_whole_year_with_norm_gps',
                             step='train', in_memory=False, acc_types=(1, 2))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10)
    for _ in tqdm(dataloader):
        pass

# Performance demonstration
if __name__ == "__main__":
    main()
