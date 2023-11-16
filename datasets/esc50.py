import os
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import pandas as pd
import librosa

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func

# specify ESC50 location in 'dataset_dir'
# 3 files have to be located there:
# - FSD50K.eval_mp3.hdf
# - FSD50K.val_mp3.hdf
# - FSD50K.train_mp3.hdf
# follow the instructions here to get these 3 files:
# https://github.com/kkoutini/PaSST/tree/main/esc50

dataset_dir = None

assert dataset_dir is not None, "Specify ESC50 dataset location in variable 'dataset_dir'. " \
                                "Check out the Readme file for further instructions. " \
                                "https://github.com/fschmid56/EfficientAT/blob/main/README.md"

dataset_config = {
    'meta_csv': os.path.join(dataset_dir, "meta/esc50.csv"),
    'audio_path': os.path.join(dataset_dir, "audio_32k/"),
    'num_of_classes': 50
}


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, f2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, f1, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class AudioSetDataset(TorchDataset):
    def __init__(self, meta_csv, audiopath, fold, train=False, resample_rate=32000, classes_num=50,
                 clip_length=5, gain_augment=0):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.resample_rate = resample_rate
        self.meta_csv = meta_csv
        self.df = pd.read_csv(meta_csv)
        if train:  # training all except this
            print(f"Dataset training fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold != fold]
            print(f" for training remains {len(self.df)}")
        else:
            print(f"Dataset testing fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold == fold]
            print(f" for testing remains {len(self.df)}")

        self.clip_length = clip_length * resample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment
        self.audiopath = audiopath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]

        waveform, _ = librosa.load(self.audiopath + row.filename, sr=self.resample_rate, mono=True)
        if self.gain_augment:
            waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length)
        target = np.zeros(self.classes_num)
        target[row.target] = 1
        return waveform.reshape(1, -1),  row.filename, target


def get_base_training_set(resample_rate=32000, gain_augment=0, fold=1):
    meta_csv = dataset_config['meta_csv']
    audiopath = dataset_config['audio_path']
    ds = AudioSetDataset(meta_csv, audiopath, fold, train=True,
                         resample_rate=resample_rate, gain_augment=gain_augment)
    return ds


def get_base_test_set(resample_rate=32000, fold=1):
    meta_csv = dataset_config['meta_csv']
    audiopath = dataset_config['audio_path']
    ds = AudioSetDataset(meta_csv, audiopath, fold, train=False, resample_rate=resample_rate)
    return ds


def get_training_set(resample_rate=32000, roll=False, wavmix=False, gain_augment=0, fold=1):
    ds = get_base_training_set(resample_rate=resample_rate, gain_augment=gain_augment, fold=fold)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_test_set(resample_rate=32000, fold=1):
    ds = get_base_test_set(resample_rate, fold=fold)
    return ds
