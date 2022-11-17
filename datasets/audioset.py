import io
import os
import av
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DistributedSampler, WeightedRandomSampler
import torch
import numpy as np
import h5py

from datasets.helpers.audiodatasets import PreprocessDataset


base_dir = "/share/rk7/shared/audioset_hdf5s/"
dataset_config = {
    'roll': False,  # apply roll augmentation
    'gain_augment': 0,
    'wavemix': False,
    'base_dir': "/share/rk7/shared/audioset_hdf5s/",  # base directory of the dataset, change it or make a link
    'balanced_train_hdf5': base_dir + "mp3/balanced_train_segments_mp3.hdf",
    'eval_hdf5': base_dir + "mp3/eval_segments_mp3.hdf",
    'unbalanced_train_hdf5': base_dir + "mp3/unbalanced_train_segments_mp3.hdf",
    'num_of_classes': 527,
    'resample_rate': 32000
}


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    # print(stream)
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


class AddIndexDataset(TorchDataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        x, f, y = self.ds[index]
        return x, f, y, index

    def __len__(self):
        return len(self.ds)


class AudioSetDataset(TorchDataset):
    def __init__(self, hdf5_file, sample_rate=32000, resample_rate=32000, classes_num=527,
                 clip_length=10, augment=False, in_mem=False):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.sample_rate = sample_rate
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
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num
        self.augment = augment
        if augment:
            print(f"Will agument data from {hdf5_file}")

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
        if self.augment:
            waveform = pydub_augment(waveform, dataset_config.get('gain_augment', 0))
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


def get_base_training_set():
    balanced_train_hdf5 = dataset_config['balanced_train_hdf5']
    resample_rate = dataset_config['resample_rate']
    ds = AudioSetDataset(balanced_train_hdf5, augment=True, resample_rate=resample_rate)
    return ds


def get_unbalanced_training_set():
    unbalanced_train_hdf5 = dataset_config['unbalanced_train_hdf5']
    resample_rate = dataset_config['resample_rate']
    ds = AudioSetDataset(unbalanced_train_hdf5, augment=True, resample_rate=resample_rate)
    return ds


def get_norms_dataset():
    unbalanced_train_hdf5 = dataset_config['unbalanced_train_hdf5']
    balanced_train_hdf5 = dataset_config['balanced_train_hdf5']
    resample_rate = dataset_config['resample_rate']
    ds = ConcatDataset(
        [AudioSetDataset(balanced_train_hdf5, augment=False, resample_rate=resample_rate),
         AudioSetDataset(unbalanced_train_hdf5, augment=False, resample_rate=resample_rate)])
    return ds


def get_base_full_training_set():
    sets = [get_base_training_set(), get_unbalanced_training_set()]
    ds = ConcatDataset(sets)
    return ds


def get_ft_cls_balanced_sample_weights(sample_weight_offset=100, sample_weight_sum=True):
    """
    :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5,unbalanced_train_hdf5 is important.
    # should match get_full_training_set
    unbalanced_train_hdf5 = dataset_config['unbalanced_train_hdf5']
    balanced_train_hdf5 = dataset_config['balanced_train_hdf5']
    num_of_classes = dataset_config['num_of_classes']

    all_y = []
    for hdf5_file in [balanced_train_hdf5, unbalanced_train_hdf5]:
        with h5py.File(hdf5_file, 'r') as dataset_file:
            target = dataset_file['target']
            target = np.unpackbits(target, axis=-1, count=num_of_classes)
            all_y.append(target)
    all_y = np.concatenate(all_y, axis=0)
    all_y = torch.as_tensor(all_y)
    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    per_class = sample_weight_offset + per_class  # offset low freq classes
    if sample_weight_offset > 0:
        print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")
    per_class_weights = 1000. / per_class
    all_weight = all_y * per_class_weights
    if sample_weight_sum:
        print("\nsample_weight_sum\n")
        all_weight = all_weight.sum(dim=1)
    else:
        all_weight, _ = all_weight.max(dim=1)
    return all_weight


def get_ft_weighted_sampler(epoch_len=100000, sampler_replace=False):
    samples_weights = get_ft_cls_balanced_sample_weights
    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    print("num_nodes= ", num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    return DistributedSamplerWrapper(sampler=WeightedRandomSampler(samples_weights,
                                                                   num_samples=epoch_len, replacement=sampler_replace),
                                     dataset=range(epoch_len),
                                     num_replicas=num_nodes,
                                     rank=rank,
                                     )


def get_base_test_set():
    eval_hdf5 = dataset_config['eval_hdf5']
    resample_rate = dataset_config['resample_rate']
    ds = AudioSetDataset(eval_hdf5, resample_rate=resample_rate)
    return ds


def get_roll_func(axis=1, shift=None, shift_range=50):
    print("rolling...")

    def roll_func(b):
        x, i, y = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        return x.roll(sf, axis), i, y
    return roll_func


def get_training_set():
    roll = dataset_config['roll']
    wavmix = dataset_config['wavemix']
    ds = get_base_training_set()
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_full_training_set(add_index=True):
    roll = dataset_config['roll']
    wavmix = dataset_config['wavemix']
    ds = get_base_full_training_set()
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    if add_index:
        ds = AddIndexDataset(ds)
    return ds


def get_test_set(add_index_test=False):
    ds = get_base_test_set()
    if add_index_test:
        ds = AddIndexDataset(ds)
    return ds


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
