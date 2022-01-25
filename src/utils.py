from math import factorial as f
import networkx as nx
import numpy as np
from random import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import config as cfg


def flip(p):
    return True if random() < p else False


def get_dataloader(data, targets, batchsize, shuffle=False):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def get_testloader(args):
    kwargs = {}
    if args.dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST(cfg.data_root, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(cfg.data_root, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST(cfg.data_root, train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


def nCr(n, r):
    return f(n)//f(r)//f(n-r)


def in_range(elem, upper, lower):
    return (elem >= lower) and (elem <= upper)


def get_spectral_radius(matrix):
    eig, _ = np.linalg.eig(matrix)

    return max(eig)


def get_average_degree(graph):
    return sum(dict(graph.degree()).values())/len(graph)


def get_max_degree(graph):
    return max(dict(graph.degree()).values())


def get_laplacian(graph):
    return nx.laplacian_matrix(graph).toarray()


def get_rho(graph, num_nodes, factor):
    max_d = get_max_degree(graph)
    d = 1/(factor*max_d)
    L = get_laplacian(graph)
    V = np.eye(num_nodes) - d*L
    Z = V-(1/num_nodes)
    return get_spectral_radius(Z)


def decimal_format(num, places=4):
    return round(num, places)
