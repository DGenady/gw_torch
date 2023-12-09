import torch
import pickle
from torch.utils.data import Dataset


class GWDataset(Dataset):

    def __init__(self, files):
        self.data_files = files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        sample_file = self.data_files[index]
        with open(sample_file, "rb") as f:
            sample = pickle.load(f)
        H_sample = torch.from_numpy(sample['H']['clean_spec']).float(), torch.zeros(1)[0]  # label is 0 for H and 1 for L
        L_sample = torch.from_numpy(sample['L']['clean_spec']).float(), torch.ones(1)[0]
        return H_sample, L_sample


class SynthDataset(Dataset):
    def __init__(self, files):
        self.data_files = files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        sample_file = self.data_files[index]
        with open(sample_file, "rb") as f:
            sample = pickle.load(f)
        H_sample_clean = torch.from_numpy(sample['H']['clean_spec']).float(), torch.zeros(1)[0]  # label is 0 for H and 1 for L
        L_sample_clean = torch.from_numpy(sample['L']['clean_spec']).float(), torch.ones(1)[0]
        H_sample_signal = torch.from_numpy(sample['H']['signal_spec']).float(), torch.zeros(1)[0]
        L_sample_signal = torch.from_numpy(sample['L']['signal_spec']).float(), torch.ones(1)[0]
        return H_sample_clean, L_sample_clean, H_sample_signal, L_sample_signal
