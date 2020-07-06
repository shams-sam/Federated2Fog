from math import factorial as f
from random import random
from torch.utils.data import TensorDataset, DataLoader


def flip(p):
    return True if random() < p else False


def get_dataloader(data, targets, batchsize, shuffle=True):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def nCr(n, r):
    return f(n)//f(r)//f(n-r)
