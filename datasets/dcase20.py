import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import librosa

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func, get_gain_augment_func

dataset_dir = None
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2020 Mobile dataset' location in variable " \
                                "'dataset_dir'. Check out the Readme file for further instructions. " \
                                "https://github.com/fschmid56/EfficientAT/blob/main/README.md"

dataset_config = {
    "dataset_name": "tau_urban_acoustic_scene20",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "train_files_csv": os.path.join(dataset_dir, "evaluation_setup", "fold1_train.csv"),
    "test_files_csv": os.path.join(dataset_dir, "evaluation_setup", "fold1_evaluate.csv")
}


class BasicDCASE20Dataset(TorchDataset):
    """
    Basic DCASE20 Dataset
    """

    def __init__(self, meta_csv, sr=32000, cache_path=None):
        """
        @param meta_csv: meta csv file for the dataset
        @param sr: specify sampling rate
        @param sr: specify cache path to store resampled waveforms
        return: waveform, name of the file, label, device and cities
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)
        self.sr = sr
        if cache_path is not None:
            self.cache_path = os.path.join(cache_path, dataset_config["dataset_name"] + f"_r{self.sr}", "files_cache")
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            self.cache_path = None

    def __getitem__(self, index):
        if self.cache_path:
            cpath = os.path.join(self.cache_path, str(index) + ".pt")
            try:
                sig = torch.load(cpath)
            except FileNotFoundError:
                sig, _ = librosa.load(os.path.join(dataset_dir, self.files[index]), sr=self.sr, mono=True)
                sig = torch.from_numpy(sig[np.newaxis])
                torch.save(sig, cpath)
        else:
            sig, _ = librosa.load(os.path.join(dataset_dir, self.files[index]), sr=self.sr, mono=True)
            sig = torch.from_numpy(sig[np.newaxis])
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)


class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for 'training', 'testing'
        return: x, label, device, city, index
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city, self.available_indices[index]

    def __len__(self):
        return len(self.available_indices)


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5, num_classes=10):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        self.num_classes = num_classes
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        x1, f1, y1, d1, c1, i1 = self.dataset[index]
        y = np.zeros(self.num_classes, dtype="float32")
        y[y1] = 1.
        y1 = y
        if torch.rand(1) < self.rate:
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, _, y2, _, _, _ = self.dataset[idx2]
            y = np.zeros(self.num_classes, dtype="float32")
            y[y2] = 1.
            y2 = y
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, f1, (y1 * l + y2 * (1. - l)), d1, c1, i1
        return x1, f1, y1, d1, c1, i1

    def __len__(self):
        return len(self.dataset)


# commands to create the datasets for training and testing
def get_training_set(cache_path=None, resample_rate=32000, roll=False, gain_augment=False, wavmix=False):
    ds = get_base_training_set(dataset_config['meta_csv'], dataset_config['train_files_csv'], cache_path,
                               resample_rate)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())

    if gain_augment:
        ds = PreprocessDataset(ds, get_gain_augment_func(gain_augment))

    if wavmix:
        ds = MixupDataset(ds)

    return ds


def get_base_training_set(meta_csv, train_files_csv, cache_path, resample_rate):
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE20Dataset(meta_csv, sr=resample_rate, cache_path=cache_path), train_indices)
    return ds


def get_test_set(cache_path=None, resample_rate=32000):
    ds = get_base_test_set(dataset_config['meta_csv'], dataset_config['test_files_csv'], cache_path,
                           resample_rate)
    return ds


def get_base_test_set(meta_csv, test_files_csv, cache_path, resample_rate):
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE20Dataset(meta_csv, sr=resample_rate, cache_path=cache_path), test_indices)
    return ds
