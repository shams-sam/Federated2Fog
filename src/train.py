from collections import OrderedDict, defaultdict
from multi_class_hinge_loss import multiClassHingeLoss
from networkx import laplacian_matrix, is_connected
from networkx.generators.geometric import random_geometric_graph
import numpy as np
from random import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def get_model_weights(model, scaling_factor=1):

    if scaling_factor == 1:
        return model.state_dict()

    else:
        weights = model.state_dict()
        for key, val in weights.items():
            weights[key] = val*scaling_factor
        return weights


def add_model_weights(weights1, weights2):

    for key, val in weights2.items():
        weights1[key] += val

    return weights1


def averaging_consensus(cluster, models, weights):
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_], weights[_])
                           for _ in cluster]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)

    return model_sum


def get_connected_graph(num_nodes, radius):
    graph = random_geometric_graph(num_nodes, radius)
    while not is_connected(graph):
        graph = random_geometric_graph(num_nodes, radius)

    return graph


def laplacian_average(models, V, num_nodes, rounds):
    model = OrderedDict()
    idx = np.random.randint(0, num_nodes)
    for key, val in models[0].items():
        size = val.size()
        initial = torch.stack([_[key] for _ in models])
        final = torch.matmul(torch.matrix_power(V, rounds),
                             initial.reshape(num_nodes, -1))*num_nodes
        model[key] = final[idx].reshape(size)

    return model


def laplacian_consensus(cluster, models, weights, device,
                        rounds, radius, factor):
    num_nodes = len(cluster)
    graph = get_connected_graph(num_nodes, radius)
    max_deg = max(dict(graph.degree()).values())
    d = 1/(factor*max_deg)
    L = laplacian_matrix(graph).toarray()
    V = torch.Tensor(np.eye(num_nodes) - d*L).to(device)
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_].get(), weights[_])
                           for _ in cluster]
        model_sum = laplacian_average(weighted_models, V, num_nodes, rounds)

    return model_sum


def flip(p):
    return True if random() < p else False


def weight_gradient(w1, w2, lr):
    return torch.norm((w1.flatten()-w2.flatten())/lr).item()


def model_gradient(model1, model2, lr):
    grads = defaultdict(list)
    for key, val in model1.items():
        grads[key] = weight_gradient(model1[key], model2[key], lr)

    return grads


def get_dataloader(data, targets, batchsize, shuffle=True):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def fog_train(args, model, fog_graph, nodes, X_trains, y_trains,
              device, epoch, loss_fn, consensus,
              rounds, radius, d2d, factor=10):
    # fog learning with model averaging

    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_models = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        worker_optims[w] = optim.SGD(
            params=node_model.parameters(), lr=args.lr,
            weight_decay=1e-5 if loss_fn=='hinge' else 0)
        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, args.batch_size)

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            worker_optims[w].zero_grad()
            output = node_model(data)
            loss = loss_fn_(output, target)
            loss.backward()
            worker_optims[w].step()
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    for l in range(1, len(args.num_clusters)+1):
        aggregators = [_ for _ in nodes.keys() if 'L{}'.format(l) in _]
        for a in aggregators:
            worker_models[a] = model.copy().send(nodes[a])
            worker_num_samples[a] = 1
            children = fog_graph[a]

            for child in children:
                worker_models[child].move(nodes[a])

            if consensus == 'averaging' or flip(1-d2d):
                model_sum = averaging_consensus(children, worker_models,
                                                worker_num_samples)
                worker_models[a].load_state_dict(model_sum)
            elif consensus == 'laplacian':
                model_sum = laplacian_consensus(children, worker_models,
                                                worker_num_samples, device,
                                                rounds, radius, factor)
                agg_model = worker_models[a].get()
                agg_model.load_state_dict(model_sum)
                worker_models[a] = agg_model.send(nodes[a])
            else:
                raise Exception

    assert len(aggregators) == 1
    master = get_model_weights(worker_models[aggregators[0]].get(),
                               1/args.num_train)

    grad = model_gradient(model.state_dict(), master, args.lr)
    model.load_state_dict(master)

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        print('Train Epoch: {} \tLoss: {:.6f} +- {:.6f} \tGrad: {}'.format(
            epoch,
            loss.mean(), loss.std(), dict(grad).values()
        ))

    return grad


def fl_train(args, model, fog_graph, nodes, X_trains, y_trains,
             device, epoch, loss_fn='nll'):
    # federated learning with model averaging

    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_models = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        worker_optims[w] = optim.SGD(
            params=worker_models[w].parameters(), lr=args.lr)

        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, args.batch_size)

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            worker_optims[w].zero_grad()
            output = node_model(data)
            loss = loss_fn_(output, target)
            loss.backward()
            worker_optims[w].step()
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    agg = 'L1_W0'
    worker_models[agg] = model.copy().send(nodes[agg])
    children = fog_graph[agg]

    for child in children:
        worker_models[child].move(nodes[agg])

    with torch.no_grad():
        weighted_models = [get_model_weights(
            worker_models[_],
            worker_num_samples[_]/args.num_train) for _ in children]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)
        worker_models[agg].load_state_dict(model_sum)

    master = get_model_weights(worker_models[agg].get())

    grad = model_gradient(model.state_dict(), master, args.lr)
    model.load_state_dict(master)

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        print('Train Epoch: {} \tLoss: {:.6f} +- {:.6f} \tGrad: {}'.format(
            epoch,
            loss.mean(), loss.std(), dict(grad).values()
        ))

    return grad


def fl_train_with_fl(args, model, device, train_loader, optimizer, epoch):
    # federated learing with federated loader
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        total += data.shape[0]
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size,
                len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))


# Test
def test(args, model, device, test_loader, best, epoch=0, loss_fn='nll'):

    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_fn == 'nll':
                test_loss += loss_fn_(output, target, reduction='sum').item()
            elif loss_fn == 'hinge':
                test_loss += loss_fn_(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if epoch % args.log_interval == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ==> '
              '{:.2f}%'.format(
                  test_loss, correct, len(test_loader.dataset),
                  100.*accuracy, 100.*best))

    return accuracy, test_loss
