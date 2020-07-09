from collections import defaultdict
from fcn import FCN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from svm import SVM


matplotlib.rcParams.update({'font.size': 14})

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

lr = 0.01
b = 32
n = 125
radius = 'graph_multi'
d2d = 1.0
non_iids = [10]
epochs = 50
factor = 2
alpha = 9e-1
rounds = 2
num_layers = 3
dyn = True
laplace = True
theta = True
delta = 0.00050000
omega = 1.4
eps_mul = 1.0001
kappa = [5, 15, 25, 35, 45]


fig_div_all = plt.figure(figsize=(20, 12))
fig_div_avg = plt.figure(figsize=(20, 12))
fig_rounds_all = plt.figure(figsize=(20, 12))
fig_rounds_avg = plt.figure(figsize=(20, 12))

title = ['', '', 'a', 'b', 'c', 'd']
for non_iid in non_iids:
    name = 'mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}'
    if laplace:
        name += '_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}'
    if theta:
        name += '_alpha_{}'
    if dyn:
        name += '_dyn_{}_delta_{:.8f}_omega_{}_eps_mul_{}_kappa_{}'
    name = name.format(
        clf, non_iid, n, lr, b, rounds, radius, d2d, factor, alpha, dyn, delta, omega, eps_mul, kappa
    )
    files = '../history/history_{}.pkl'.format(name)
    x_ax, y_ax, l_test, grad_tr, rounds_tr, div_tr = pkl.load(open(files, 'rb'))
    x_ax = list(range(epochs))

    colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']
    colors_ = ['r', 'm', 'b', 'g', 'c', 'k']

    plt_idx = 1
    for l in range(num_layers):
        ax_div_all = fig_div_all.add_subplot(220 + plt_idx)
        d_ax = defaultdict(list)
        d_all = defaultdict(list)
        r_ax = defaultdict(list)
        r_all = defaultdict(list)
        d_1 = defaultdict(list)
        r_1 = defaultdict(list)

        for div in div_tr:
            d_ax[l].append(sum(div[l])/len(div[l]))
            for idx in range(len(div[l])):
                d_all[idx].append(div[l][idx])
        for idx in range(0, len(div[l])):
            ax_div_all.plot(x_ax, d_all[idx][:epochs], label='div @L'+str(idx))
        ax_div_avg = fig_div_avg.add_subplot(220 + plt_idx)
        ax_div_avg.plot(x_ax, d_ax[l][:epochs], colors_[l], label='div @L'+str(l))
        ax_div_avg.legend()

        if theta:
            ax_rounds_all = fig_rounds_all.add_subplot(220 + plt_idx)
            for r in rounds_tr:
                r_ax[l].append(sum(r[l])/len(r[l]))
                for idx in range(len(r[l])):
                    r_all[idx].append(r[l][idx])
            for idx in range(0, len(r[l])):
                ax_rounds_all.plot(x_ax, r_all[idx][:epochs], label='rounds @L'+str(idx))

            ax_rounds_avg = fig_rounds_avg.add_subplot(220 + plt_idx)
            ax_rounds_avg.plot(x_ax, r_ax[l][:epochs], colors_[l], label='rounds @L'+str(l))
            ax_rounds_avg.set_xlabel('epochs')
            ax_rounds_avg.set_ylabel('rounds')
            ax_rounds_avg.set_title('({})'.format(title[plt_idx]), y=-0.22)
            ax_rounds_avg.legend()
        plt_idx += 1

    fig_name = '../plots/_{}_div_all.png'.format(name)
    print('Saving: ', fig_name)
    fig_div_all.subplots_adjust(wspace=0.3)
    fig_div_all.savefig(fig_name)

    fig_name = '../plots/_{}_div_avg.png'.format(name, l)
    print('Saving: ', fig_name)
    fig_div_avg.subplots_adjust(wspace=0.3)
    fig_div_avg.savefig(fig_name)

    if theta:
        fig_name = '../plots/_{}_rounds_all.png'.format(name, l)
        print('Saving: ', fig_name)
        fig_rounds_all.subplots_adjust(wspace=0.3)
        fig_rounds_all.savefig(fig_name)

        fig_name = '../plots/_{}_rounds_avg'.format(name, l)
        print('Saving: ', fig_name)
        fig_rounds_avg.subplots_adjust(wspace=0.1, hspace=0.3)
        for format_ in ['png', 'eps']:
            fig_rounds_avg.savefig(fig_name + '.' + format_,
                                   bbox_inches='tight',
                                   dpi=300,
                                   format=format_)
