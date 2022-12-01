from torch.utils.data import Dataset
import torch
import numpy as np


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


def get_roll_func(axis=1, shift=None, shift_range=10000):
    # roll waveform (over time)
    def roll_func(b):
        x = b[0]
        others = b[1:]
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        return (x.roll(sf, axis), *others)
    return roll_func
