from consensus import averaging_consensus, consensus_matrix, \
    estimate_true_gradient, laplacian_consensus, \
    get_alpha, get_alpha_closed_form, get_cluster_eps, get_true_cluster_eps
from model_op import add_model_weights, get_model_weights, \
    get_num_params, model_gradient
from multi_class_hinge_loss import multiClassHingeLoss
import numpy as np
from random import shuffle
from terminaltables import AsciiTable
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import flip, get_dataloader


def fog_train(args, model, fog_graph, nodes, X_trains, y_trains,
              device, epoch, loss_fn, consensus,
              rounds, radius, d2d, factor=10,
              alpha_store={}, prev_grad=0, shuffle_worker_data=True):
    # fog learning with model averaging
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    log = []
    log_head = []
    if args.var_theta:
        if args.true_eps:
            log_head.append('est')
        log_head += ['div', 'true_grad']
        if args.dynamic_alpha:
            log_head += ['delta', 'D', 'mu', 'delta',
                         'eta', 'grad', 'omega', 'N',
                         'L', 'sig(c)', 'phi']
        log_head += ['rounds', 'agg', 'rho', 'sig', 'cls_n']
        log_head.append('rounded')
    log.append(log_head)

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_models = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker
    if shuffle_worker_data:
        data = list(zip(X_trains, y_trains))
        shuffle(data)
        X_trains, y_trains = zip(*data)

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        worker_optims[w] = optim.SGD(
            params=node_model.parameters(),
            lr=args.lr*(1-args.lr/10)**epoch if args.nesterov else args.lr,
            weight_decay=args.decay if loss_fn == 'hinge' else 0,)
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

    num_rounds = []
    num_div = []
    var_radius = type(radius) == list
    for l in range(1, len(args.num_clusters)+1):

        aggregators = [_ for _ in nodes.keys() if 'L{}'.format(l) in _]
        N = len(aggregators)
        cluster_rounds = []
        cluster_div = []
        for a in aggregators:
            agg_log = []

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
                num_nodes_in_cluster = len(children)
                V = consensus_matrix(num_nodes_in_cluster,
                                     radius if not var_radius else radius[l-1],
                                     factor, args.topology)
                eps = get_cluster_eps(children, worker_models, nodes, fog_graph)
                if args.true_eps:
                    est_eps = eps
                    agg_log.append(est_eps)
                    eps = get_true_cluster_eps(
                        children, worker_models, nodes, fog_graph)
                    print(eps, est_eps)
                agg_log.append(eps)
                cluster_div.append(eps)

                if args.var_theta:
                    Z = V - (1/num_nodes_in_cluster)
                    eig, dump = np.linalg.eig(Z)
                    lamda = eig.max()
                    if lamda == 0:
                        rounds = args.rounds
                    else:
                        true_grad = estimate_true_gradient(
                            prev_grad, args.omega)
                        agg_log.append(true_grad)
                        if args.dynamic_alpha and true_grad:
                            phi = sum(args.num_clusters)
                            L = len(args.num_clusters)+1
                            num_params = get_num_params(model)
                            alpha, alph_log = get_alpha_closed_form(
                                args, prev_grad, phi, N, L, num_params)
                            agg_log += alph_log
                            agg_log += [alpha, phi]
                        else:
                            alpha_store = get_alpha(
                                num_nodes_in_cluster,
                                eps, a, alpha_store,
                                args.alpha, args.dynamic_alpha)
                            alpha = alpha_store[a]
                        rounds = (np.log2(alpha)-2*np.log2(
                                (num_nodes_in_cluster**2)*eps
                        ))/(2*np.log2(lamda))
                        agg_log += [rounds, a, lamda,
                                    alpha, num_nodes_in_cluster]
                        try:
                            rounds = int(np.ceil(rounds))
                        except TypeError:
                            rounds = 50
                        if rounds > 50:
                            rounds = 50
                        elif rounds < 1:
                            rounds = 1
                    cluster_rounds.append(rounds)
                    agg_log.append(rounds)
                model_sum = laplacian_consensus(children, worker_models,
                                                worker_num_samples,
                                                V.to(device), rounds)
                agg_model = worker_models[a].get()
                agg_model.load_state_dict(model_sum)
                worker_models[a] = agg_model.send(nodes[a])
            else:
                raise Exception
            print(agg_log)
            log.append(agg_log)

        num_rounds.append(cluster_rounds)
        num_div.append(cluster_div)

    table = AsciiTable(log)
    print(table.table)
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

    return grad, num_rounds, num_div, alpha_store


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
