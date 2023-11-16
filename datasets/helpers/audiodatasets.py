from torch.utils.data import Dataset
import torch
import numpy as np
from functools import partial


class PreprocessDataset(Dataset):
    """A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def get_roll_func(axis=1, shift=None, shift_range=4000):
    return partial(roll_func, axis=axis, shift=shift, shift_range=shift_range)


# roll waveform (over time)
def roll_func(b, axis=1, shift=None, shift_range=4000):
    x = b[0]
    others = b[1:]
    x = torch.as_tensor(x)
    sf = shift
    if shift is None:
        sf = int(np.random.random_integers(-shift_range, shift_range))
    return (x.roll(sf, axis), *others)


def get_gain_augment_func(gain_augment):
    return partial(gain_augment_func, gain_augment=gain_augment)


def gain_augment_func(b, gain_augment=12):
    x = b[0]
    others = b[1:]
    gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
    amp = 10 ** (gain / 20)
    x = x * amp
    return (x, *others)
