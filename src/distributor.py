from collections import defaultdict
from networkx import is_connected
from networkx.generators.geometric import random_geometric_graph
from networkx.generators.random_graphs import erdos_renyi_graph
import numpy as np
import random
from sklearn.model_selection import train_test_split
import syft as sy
import torch


def get_cluster_sizes(total, num_parts, uniform=True):
    parts = []
    if uniform:
        part_size = total//num_parts
        parts = [part_size] * num_parts
        for _ in range(total - sum(parts)):
            parts[2*_] += 1
    else:
        crossover = [0] + \
                    list(sorted(random.sample(range(2, total), num_parts-1))) \
                    + [total]
        for i in range(1, len(crossover)):
            parts.append(crossover[i]-crossover[i-1])

    return parts


def assign_classes(classes, num_nodes, n):
    num_stacks = n*num_nodes
    class_stack = []
    num_classes = len(classes)
    classes = classes[np.random.permutation(num_classes)]
    i = 0
    while len(class_stack) < num_stacks:
        class_stack.append(classes[i])
        i += 1
        if i == len(classes):
            i = 0
            classes = classes[np.random.permutation(num_classes)]

    class_stack = [sorted(class_stack[i*n: i*n + n])
                   for i in range(num_nodes)]

    class_map = defaultdict(list)
    for node_id in range(len(class_stack)):
        for _ in range(n):
            class_map[class_stack[node_id][_]].append(node_id)
    return class_stack, class_map


def get_distributed_data(X_train, y_train, num_parts,
                         stratify=True, repeat=False,
                         uniform=True, shuffle=True,
                         non_iid=10, num_classes=10):
    if shuffle:
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]

    X_trains = []
    y_trains = []
    if non_iid == num_classes:
        for i in range(num_parts-1):
            if uniform:
                test_size = 1/(num_parts-i)
            else:
                test_size = 0.1
            if stratify:
                X_train, X_iter, y_train, y_iter = train_test_split(
                    X_train, y_train, stratify=y_train, test_size=test_size)
            else:
                X_train, X_iter, y_train, y_iter = train_test_split(
                    X_train, y_train, test_size=test_size)

            X_trains.append(X_iter)
            y_trains.append(y_iter)

        X_trains.append(X_train)
        y_trains.append(y_train)
    else:
        X_train_class = {}
        y_train_class = {}
        for cls in range(10):
            indices = torch.where(y_train == cls)
            X_train_class[cls] = X_train[indices]
            y_train_class[cls] = y_train[indices]

        _, class_map = assign_classes(np.unique(y_train), num_parts, non_iid)

        X_trains = [[] for _ in range(num_parts)]
        y_trains = [[] for _ in range(num_parts)]

        for cls, node_list in class_map.items():
            X_cls = X_train_class[cls]
            y_cls = y_train_class[cls]
            num_splits = len(node_list)
            if uniform:
                split_size = X_cls.shape[0]//num_splits
                crossover = [_*split_size
                             for _ in range(num_splits)] + [X_cls.shape[0]]
            else:
                crossover = [0] + \
                    list(sorted(random.sample(
                        range(2, X_cls.shape[0]), num_splits-1))) \
                    + [X_cls.shape[0]]
            for id_ in range(len(node_list)):
                X_trains[node_list[id_]].append(
                    X_cls[crossover[id_]: crossover[id_+1]])
                y_trains[node_list[id_]].append(
                    y_cls[crossover[id_]: crossover[id_+1]])

        for id_ in range(num_parts):
            X_trains[id_] = torch.cat(X_trains[id_], dim=0)
            y_trains[id_] = torch.cat(y_trains[id_], dim=0)
    assert len(X_trains) == num_parts

    return X_trains, y_trains


def get_distributed_data_using_loader(train_loader):
    X_trains = []
    y_trains = []

    for batch_idx, (data, target) in enumerate(train_loader):
        X_trains.append(data)
        y_trains.append(target)

    return X_trains, y_trains


def get_fog_graph(hook, num_workers, num_clusters,
                  shuffle_workers=True, uniform_clusters=True, fog=True):
    # Define workers and layers
    workers = {}
    agg_map = {}
    layer = 0
    for id_ in range(num_workers):
        name = 'L{}_W{}'.format(layer, id_)
        workers[name] = sy.VirtualWorker(hook, id=name)

    layer = 1

    if not fog:
        # single layer model averaging fl
        name = 'L1_W0'
        workers[name] = sy.VirtualWorker(hook, id=name)
        worker_ids = [_ for _ in workers.keys() if 'L0' in _]
        agg_map[name] = worker_ids

        return agg_map, workers

    for num_cluster in num_clusters:
        # multi layer aggregation fog learning
        for id_ in range(num_cluster):
            name = 'L{}_W{}'.format(layer, id_)
            workers[name] = sy.VirtualWorker(hook, id=name)
        layer += 1

    for l in range(1, len(num_clusters)+1):
        clustr_ids = [_ for _ in workers.keys() if 'L{}'.format(l) in _]
        worker_ids = [_ for _ in workers.keys() if 'L{}'.format(l-1) in _]
        if shuffle_workers:
            worker_ids = list(np.array(worker_ids)[
                np.random.permutation(len(worker_ids))])
        cluster_sizes = get_cluster_sizes(len(worker_ids),
                                          len(clustr_ids), uniform_clusters)
        indices = [sum(cluster_sizes[:id_])
                   for id_ in range(len(cluster_sizes)+1)]
        for id_ in range(len(clustr_ids)):
            agg_map[clustr_ids[id_]] = worker_ids[indices[id_]: indices[id_+1]]

    return agg_map, workers


def get_connected_graph(num_nodes, param, topology='rgg', retries=10):
    if topology == 'rgg':
        generator = random_geometric_graph
    elif topology == 'er':
        generator = erdos_renyi_graph
    graph = generator(num_nodes, param)
    counter = 0
    while not is_connected(graph):
        graph = generator(num_nodes, param)
        counter += 1
        if counter > retries:
            return False

    return graph
