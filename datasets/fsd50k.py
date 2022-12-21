import io
import os
import av
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler
import torch
import numpy as np
import h5py

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func

# specify FSD50K location in 'dataset_dir'
# 3 files have to be located there:
# - FSD50K.eval_mp3.hdf
# - FSD50K.val_mp3.hdf
# - FSD50K.train_mp3.hdf
# follow the instructions here to get these 3 files:
# https://github.com/kkoutini/PaSST/tree/main/fsd50k

dataset_dir = None

assert dataset_dir is not None, "Specify FSD50K dataset location in variable 'dataset_dir'. " \
                                "Check out the Readme file for further instructions. " \
                                "https://github.com/fschmid56/EfficientAT/blob/main/README.md"

dataset_config = {
    'balanced_train_hdf5': os.path.join(dataset_dir, "FSD50K.train_mp3.hdf"),
    'valid_hdf5': os.path.join(dataset_dir, "FSD50K.val_mp3.hdf"),
    'eval_hdf5': os.path.join(dataset_dir, "FSD50K.eval_mp3.hdf"),
    'num_of_classes': 200
}


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if audio_length is None:
        # audio_length not specified don't do anything.
        return x
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        offset = torch.randint(0, len(x) - audio_length + 1, (1,)).item()
        return x[offset:offset + audio_length]


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
    def __init__(self, hdf5_file, resample_rate=32000, classes_num=200, clip_length=10,
                 in_mem=False, gain_augment=0):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.resample_rate = resample_rate
        self.hdf5_file = hdf5_file
        if in_mem:
            print("\nPreloading in memory\n")
            with open(hdf5_file, 'rb') as f:
                self.hdf5_file = io.BytesIO(f.read())
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['audio_name'])
            print(f"Dataset from {hdf5_file} with length {self.length}.")
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length
        if clip_length is not None:
            self.clip_length = clip_length * resample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None

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
        if self.dataset_file is None:
            self.open_hdf5()

        audio_name = self.dataset_file['audio_name'][index].decode()
        waveform = decode_mp3(self.dataset_file['mp3'][index])
        waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = self.resample(waveform)
        target = self.dataset_file['target'][index]
        target = np.unpackbits(target, axis=-1,
                               count=self.classes_num).astype(np.float32)
        return waveform.reshape(1, -1), audio_name, target

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.resample_rate == 32000:
            return waveform
        elif self.resample_rate == 16000:
            return waveform[0:: 2]
        elif self.resample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


def get_base_training_set(resample_rate=32000, gain_augment=0):
    balanced_train_hdf5 = dataset_config['balanced_train_hdf5']
    ds = AudioSetDataset(balanced_train_hdf5, resample_rate=resample_rate, gain_augment=gain_augment)
    return ds


def get_base_eval_set(resample_rate=32000, variable_eval=None):
    eval_hdf5 = dataset_config['eval_hdf5']
    if variable_eval:
        print("Variable length eval!!")
        ds = AudioSetDataset(eval_hdf5, resample_rate=resample_rate, clip_length=None)
    else:
        ds = AudioSetDataset(eval_hdf5, resample_rate=resample_rate)
    return ds


def get_base_valid_set(resample_rate=32000, variable_eval=None):
    valid_hdf5 = dataset_config['valid_hdf5']
    if variable_eval:
        print("Variable length valid_set !!")
        ds = AudioSetDataset(valid_hdf5, resample_rate=resample_rate, clip_length=None)
    else:
        ds = AudioSetDataset(valid_hdf5, resample_rate=resample_rate)
    return ds


def get_training_set(roll=False, wavmix=False, gain_augment=0, resample_rate=32000):
    ds = get_base_training_set(resample_rate=resample_rate, gain_augment=gain_augment)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_valid_set(resample_rate=32000, variable_eval=None):
    ds = get_base_valid_set(resample_rate, variable_eval)
    return ds


def get_eval_set(resample_rate=32000, variable_eval=None):
    ds = get_base_eval_set(resample_rate, variable_eval)
    return ds
