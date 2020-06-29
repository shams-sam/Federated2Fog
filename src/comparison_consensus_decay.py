from collections import defaultdict
from fcn import FCN
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from svm import SVM

model = SVM()
params = model.state_dict().keys()
clf = 'svm'
# lr = 0.01
# b = 300
# n = 200
# radius = 0.7
# d2d = 1.0
# non_iids = [1]
# epochs = 25
# factor = 2
# alpha = 20
# rounds = 2
# num_layers = 4

lr = 0.001
b = 32
n = 500
radius = 0.7
d2d = 1.0
non_iids = [10]
epochs = 100
factor = 2
alpha = False
rounds = 2
num_layers = 4
dyn = False
laplace = True
theta = True

for non_iid in non_iids:
    name = 'mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}'
    if laplace:
        name += '_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}'
    if theta:
        name += '_alpha_{}_dyn_{}'
    name = name.format(
        clf, non_iid, n, lr, b, rounds, radius, d2d, factor, alpha, dyn
    )
    files = '../history/history_{}.pkl'.format(name)
    x_ax, y_ax, l_test, grad_tr, rounds_tr, div_tr = pkl.load(open(files, 'rb'))
    x_ax = list(range(epochs))

    colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']
    colors_ = ['r', 'm', 'b', 'g', 'c', 'k']

    for l in range(num_layers):
        d_ax = defaultdict(list)
        d_all = defaultdict(list)
        r_ax = defaultdict(list)
        r_all = defaultdict(list)
        d_1 = defaultdict(list)
        r_1 = defaultdict(list)

        fig, ax1 = plt.subplots()
        for div in div_tr:
            d_ax[l].append(sum(div[l])/len(div[l]))
            for idx in range(len(div[l])):
                d_all[idx].append(div[l][idx])
        for idx in range(0, len(div[l])):
            plt.plot(x_ax, d_all[idx][:epochs], label='div @L'+str(idx))

        fig_name = '../plots/{}_L{}_div_all.png'.format(name, l)
        print('Saving: ', fig_name)
        plt.savefig(fig_name)

        fig, ax1 = plt.subplots()
        ax1.plot(x_ax, d_ax[l][:epochs], colors_[l], label='div @L'+str(l))
        plt.legend()
        fig_name = '../plots/{}_L{}_div_avg.png'.format(name, l)
        print('Saving: ', fig_name)
        plt.savefig(fig_name)

        if theta:
            fig, ax1 = plt.subplots()
            for r in rounds_tr:
                r_ax[l].append(sum(r[l])/len(r[l]))
                for idx in range(len(r[l])):
                    r_all[idx].append(r[l][idx])
            for idx in range(0, len(r[l])):
                plt.plot(x_ax, r_all[idx][:epochs], label='rounds @L'+str(idx))

            fig_name = '../plots/{}_L{}_rounds_all.png'.format(name, l)
            print('Saving: ', fig_name)
            plt.savefig(fig_name)

            fig, ax1 = plt.subplots()
            ax1.plot(x_ax, r_ax[l][:epochs], colors_[l], label='div @L'+str(l))
            plt.legend()
            fig_name = '../plots/{}_L{}_rounds_avg.png'.format(name, l)
            print('Saving: ', fig_name)
            plt.savefig(fig_name)
