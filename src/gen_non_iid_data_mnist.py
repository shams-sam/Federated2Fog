from arguments import Arguments
from distributor import get_distributed_data
import numpy as np
import pickle as pkl
import torch
from torchvision import datasets, transforms


args = Arguments()

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.num_train, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.num_test, shuffle=True, **kwargs)

for data, target in train_loader:
    X_train = data
    y_train = target

for data, target in test_loader:
    X_test = data
    y_test = target


def repeat_data(data, repeat=args.repeat):
    rep = [data for _ in range(repeat)]
    rep = torch.cat(rep, dim=0)

    return rep


X_train, y_train = repeat_data(X_train), repeat_data(y_train)

print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))

print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))


for non_iid in range(1, 5):
    X_trains, y_trains = get_distributed_data(X_train, y_train,
                                              args.num_workers,
                                              stratify=args.stratify,
                                              uniform=args.uniform_data,
                                              shuffle=args.shuffle_data,
                                              non_iid=non_iid)

    for _ in y_trains:
        print(np.bincount(_))

    name = ['data', 'mnist',
            'non_iid', str(non_iid),
            'num_workers', str(args.num_workers),
            'stratify', str(args.stratify),
            'uniform', str(args.uniform_data),
            'repeat', str(args.repeat),
    ]

    filename = '../ckpts/' + '_'.join(name) + '.pkl'
    print('Saving: {}'.format(filename))
    pkl.dump((X_trains, y_trains), open(filename, 'wb'))
