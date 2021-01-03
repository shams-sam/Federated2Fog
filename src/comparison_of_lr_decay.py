import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 37})
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 10


fig = plt.figure(figsize=(30, 7))

epochs = 50

dataset = 'fmnist'
factor = 32
non_iid = 1
n = 125 #625
b = 480 #96
radius = 'graph_multi'

idx = 1
title = ['', '', 'a', 'b']
for line in ['accuracy', 'loss']:
    for non_iid in [1]:
        ax = fig.add_subplot(120 + idx)
        idx += 1
        # files = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_False_batch_{}_laplace_rounds_1_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, radius, factor)
        # file_ = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_True_batch_{}_laplace_rounds_1_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, radius, factor)
        # file_fl = '../history/history_{}_svm_fl_uniform_non_iid_1_num_workers_{}_lr_0.01_decay_0.1_batch_{}_repeat_1.pkl'.format(dataset, n, b)
        # file_15 = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_False_batch_{}_laplace_rounds_15_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, radius, factor)

        files = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_False_batch_{}_laplace_rounds_1_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, 0.6, factor)
        file_ = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_True_batch_{}_laplace_rounds_1_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, 0.6, factor)
        file_fl = '../history/history_{}_svm_fl_uniform_non_iid_1_num_workers_{}_lr_0.01_decay_0.1_batch_{}_repeat_1.pkl'.format(dataset, n, b)
        file_15 = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_nest_False_batch_{}_laplace_rounds_15_radius_{}_d2d_1.0_factor_{}_repeat_1.pkl'.format(dataset, n, b, 0.6, factor)

        colors = ['r.:', 'k.:']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_fl, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, 'k*-', label='EUT')
        else:
            ax.plot(x_ax, np.array(l_test), 'k*-', label='EUT')

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_15, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, 'g.-', label='num cons = 15')
        else:
            ax.plot(x_ax, np.array(l_test), 'g.-', label=r'$\theta$ = 15')

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, 'ro:', label='num cons = 1 with decaying lr')
        else:
            ax.plot(x_ax, np.array(l_test), 'ro:', label=r'$\theta$ = 1 w/ decreasing step')

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(files, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line == 'accuracy':
            plt.plot(x_ax, y_ax, 'bs:', label='num cons = 1 w/o decaying lr')
        else:
            plt.plot(x_ax, np.array(l_test), 'bs:', label=r'$\theta$ = 1 w/o decreasing step')
        ax.set_xlabel('$k$')
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=epochs)
        plt.title('({})'.format(title[idx]), y=-0.35)
        if idx == 3:
            ax.legend(loc='upper right', ncol=4, bbox_to_anchor=(-1.25, 1.1, 2.25, .1), mode='expand', frameon=False)
file_name = '../plots/{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_batch_{}_laplace_factor_{}'.format(dataset, n, b, factor)

print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.25)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
