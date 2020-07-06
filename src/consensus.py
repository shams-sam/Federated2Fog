from collections import OrderedDict
from distributor import get_connected_graph
from model_op import get_model_weights, add_model_weights
from networkx import laplacian_matrix
import numpy as np
import torch
from utils import nCr


def averaging_consensus(cluster, models, weights):
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_], weights[_])
                           for _ in cluster]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)

    return model_sum


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


def consensus_matrix(num_nodes, radius, factor, topology):
    graph = get_connected_graph(num_nodes, radius, topology)
    max_deg = max(dict(graph.degree()).values())
    d = 1/(factor*max_deg)
    L = laplacian_matrix(graph).toarray()
    V = torch.Tensor(np.eye(num_nodes) - d*L)

    return V


# when consensus is done using d2d
# this gives closed form expression of such communication
def laplacian_consensus(cluster, models, weights, V, rounds):
    num_nodes = len(cluster)
    with torch.no_grad():
        weighted_models = [get_model_weights(models[_].get(), weights[_])
                           for _ in cluster]
        model_sum = laplacian_average(weighted_models, V, num_nodes, rounds)

    return model_sum


def get_cluster_eps(cluster, models, nodes, fog_graph,
                    param='weight', normalize=False):
    cluster_norms = []
    for _ in cluster:
        model = models[_].get()
        weight = [val for _, val in model.state_dict().items()
                  if param in _][0]
        num_childs = 1
        if _ in fog_graph:
            num_childs = len(fog_graph[_])
        norm = torch.norm(weight.flatten()).item()
        if normalize:
            norm /= num_childs
        cluster_norms.append(norm)
        models[_] = model.copy().send(nodes[_])
    cluster_norms = np.array(cluster_norms)
    cluster_norms = cluster_norms
    eps = cluster_norms.max()-cluster_norms.min()

    return eps


def get_true_cluster_eps(cluster, models, nodes, fog_graph,
                         param='weight', normalize=False):
    tuple_norms = []
    num_nodes = len(cluster)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            node_i, node_j = cluster[i], cluster[j]
            m_i, m_j = models[node_i].get(), models[node_j].get()
            w_i = [val for _, val in m_i.state_dict().items()
                   if param in _][0]
            w_j = [val for _, val in m_j.state_dict().items()
                   if param in _][0]
            tuple_norms.append(
                torch.norm(
                    w_i.flatten()-w_j.flatten()).item())
            models[node_i] = m_i.copy().send(nodes[node_i])
            models[node_j] = m_j.copy().send(nodes[node_j])

    assert len(tuple_norms) == nCr(num_nodes, 2)

    return max(tuple_norms)


def get_alpha(num_nodes, eps, a, alpha_store, mul=0.0001, dynamic=False):
    if a not in alpha_store or dynamic:
        alpha_store[a] = mul*(num_nodes**4)*(eps**2)

    return alpha_store


def estimate_true_gradient(norm_F_prev, omega):
    return 1/omega*norm_F_prev


def get_alpha_closed_form(args, grad, phi, N_prev, L, omega):
    eta = args.eta
    mu = args.decay
    if args.dynamic_delta:
        delta = estimate_delta(args)
    else:
        delta = args.delta
    print('Delta: ', delta)
    D = args.num_train
    log = [D, mu, delta, eta, grad, omega, N_prev, L]
    alpha = (D**2) * mu * (mu-delta*eta)*(grad**2)/(
        (eta**4)*phi*(omega**2)*N_prev*L)

    return alpha, log


def estimate_delta(args):
    delta = 1 - (args.eps_multiplier*(
        1-(args.decay/args.eta))**args.kappa)**(1/args.kappa)

    return delta
