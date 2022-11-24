def NAME_TO_WIDTH(name):
    map = {
        'mn04': 0.4,
        'mn05': 0.5,
        'mn10': 1.0,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0
    }
    try:
        w = map[name[:4]]
    except:
        w = 1.0

    return w


import csv

# Load label
with open('metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)


import numpy as np


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)
    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return wrapper


import torch


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    # data = data * lam + data2 * (1 - lam)
    # targets = targets * lam + targets2 * (1 - lam)
    return rn_indices, lam
