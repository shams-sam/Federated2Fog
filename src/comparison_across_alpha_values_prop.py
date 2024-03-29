import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 42})
matplotlib.rcParams['lines.linewidth'] = 4.0
matplotlib.rcParams['lines.markersize'] = 16

dataset = 'mnist'
clf = 'svm'
lr = 0.01
decay = 0.1
decay = '_decay_{}'.format(decay) if decay else ''
b = 480
n = 125
radius = 1.0
d2d = 1.0
non_iids = [1]
alpha = [1e-2, 1e-1, 6e-1, 9e-1]
epochs = 25# 50
factor = 2
rounds = 2
radius = 'graph_multi'
d2d = 1.0
num_layer = 3


rows = 1
cols = 2
fig = plt.figure(figsize=(30, 8*rows))
plt_start = rows * 100 + cols * 10
idx = 1
title = ['', '', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
for line in ['accuracy', 'rounds']:
    for non_iid in non_iids:
        plt_num = plt_start + idx
        ax = fig.add_subplot(plt_num)
        idx += 1
        files = '../history/history_{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}{}_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}_alpha_{}.pkl'
        file_ = '../history/history_{}_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}{}_batch_{}_repeat_1.pkl'.format(
            dataset, clf, non_iid, n, lr, decay, b
        )

        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, colors[-1], label='EUT')
        for i in range(len(alpha)):
            x_ax, y_ax, l_test, grad_tr, rounds_tr, _ = pkl.load(
                open(files.format(dataset, clf, non_iid, n, lr, decay, b,
                                  rounds, radius, d2d, factor, alpha[i]
                ), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                ax.plot(x_ax, y_ax, colors[i], label="$\sigma'$ = {:0.0f}%".format(alpha[i]*100))
            elif line == 'rounds':
                avg_rounds = []
                for r in rounds_tr:
                    r = np.array([_ for layer in r for _ in layer])
                    avg_rounds.append(r.sum()/len(r))
                ax.plot(x_ax, avg_rounds[:epochs], colors[i])
        if line == 'rounds':
            line = 'average \nconsensus rounds'
        ax.set_xlabel('global aggregations ($k$)')
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=25)
        ax.set_title('({})'.format(title[idx]), y=-0.35)
        if idx == 2:
            ax.legend(loc='upper right', ncol=5, bbox_to_anchor=(0, 1.1, 2.25, .1), mode='expand', frameon=False)

file_name = '../plots/{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}{}_batch_{}_laplace_alphas_{}_radius_{}_d2d_{}_factor_{}_alpha_proposal'.format(
       dataset, clf, non_iid, n, lr, decay, b, '_'.join(list(map(str, alpha))), radius, d2d, factor
)
print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.3, hspace=0.4)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
