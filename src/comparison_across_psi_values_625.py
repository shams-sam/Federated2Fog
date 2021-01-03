import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 39})
matplotlib.rcParams['lines.linewidth'] = 4.0
matplotlib.rcParams['lines.markersize'] = 16

dataset = 'fmnist'
clf = 'fcn'
lr = 0.01
b = 96
n = 625
radius = 1.0
d2d = 1.0
non_iids = [10]
exponent = 2
alpha = 9e-1
psis = [1e5, 1e4, 1e3, 1e2, 1e1]
epochs = 25
factor = 2
rounds = 2
radius = 'graph_multi'
d2d = 1.0
num_layer = 4
dyn = True
omega = 1.1

rows = 3
cols = 3
fig = plt.figure(figsize=(10*cols, 4*rows))
grid = plt.GridSpec(2, 12, hspace=0.6, wspace=10)
axs = [
    '', '',
    fig.add_subplot(grid[0, 0:4]),
    fig.add_subplot(grid[0, 4:8]),
    fig.add_subplot(grid[0, 8:12]),
    fig.add_subplot(grid[1, 0:3]),
    fig.add_subplot(grid[1, 3:6]),
    fig.add_subplot(grid[1, 6:9]),
    fig.add_subplot(grid[1, 9:12]),
]
plt_start = rows * 100 + cols * 10
idx = 1
title = ['', '', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
for line in ['accuracy', 'loss', 'rounds'] + \
    ['layer_{}'.format(_) for _ in range(num_layer)]:
    for non_iid in non_iids:
        plt_num = plt_start + idx
        idx += 1
        ax = axs[idx] 
        files = '../history/history_{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}_alpha_{}_dyn_{}_psi_{}.pkl'
        file_ = '../history/history_{}_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_repeat_1.pkl'.format(
            dataset, clf, non_iid, n, lr, b
        )

        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, np.array(y_ax), colors[-1])
        elif line=='loss':
            ax.plot(x_ax, np.array(l_test)*(10**exponent), colors[-1], label='EUT')

        for i in range(len(psis)):
            x_ax, y_ax, l_test, grad_tr, rounds_tr, _ = pkl.load(
                open(files.format(dataset, clf, non_iid, n, lr,
                                  b, rounds, radius, d2d, factor, alpha,
                                  dyn, psis[i]), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                ax.plot(x_ax, np.array(y_ax), colors[i],
                        label='psi = ' + str(psis[i]))
                # ax.set_yticks([0.75, 0.80])
            elif line == 'loss':
                label = '$\psi$ = {0:.0g}'.format(psis[i]) \
                                  if psis[i] > 100 \
                                     else '$\psi$ = {0:.0f}'.format(psis[i])
                ax.plot(x_ax, np.array(l_test)*(10**exponent),
                        colors[i], label=label)
            elif line == 'rounds':
                avg_rounds = []
                for r in rounds_tr:
                    r = np.array([_ for layer in r for _ in layer])
                    avg_rounds.append(r.sum()/len(r))
                ax.plot(x_ax, avg_rounds[:epochs], colors[i],
                        label='$\psi$ = {:0.1g}'.format(psis[i]))
            elif 'layer' in  line:
                l_num = int(line.split('_')[-1])
                layer_rounds = [[] for _ in range(num_layer)]
                for r in rounds_tr:
                    for r_idx in range(len(r)):
                        layer_rounds[r_idx].append(sum(r[r_idx])/len(r[r_idx]))

                ax.plot(x_ax, layer_rounds[l_num][:epochs],
                        colors[i], label=line)
                        
        ax.set_xlabel('$k$')
        if line == 'layer_0':
            line = r'$\overline{\theta^{(k)}_{{L}_4}}$'
        elif line=='layer_1':
            line = r'$\overline{\theta^{(k)}_{{L}_3}}$'
        elif line =='layer_2':
            line = r'$\overline{\theta^{(k)}_{{L}_2}}$'
        elif line =='layer_3':
            line = r'$\overline{\theta^{(k)}_{{L}_1}}$'
        elif line == 'rounds':
            line = r'$\overline{\theta^{(k)}}$'
        elif line == 'loss' and exponent:
            line = r'loss ($\times 10^{-2}$)'
        ax.set_ylabel(line)
        ax.set_xlim(left=0, right=25)
        ax.grid(True)
        # ax.set_xlim(left=0, right=50)
        ax.set_title('({})'.format(title[idx]), y=-0.55)
        if idx == 3:
            ax.legend(loc='upper right', ncol=7, handlelength=1.5, bbox_to_anchor=(-1.35, 1.1, 3.7, .25), mode='expand', frameon=False)
file_name = '../plots/{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}' \
            '_batch_{}_laplace_alpha_{}_radius_{}_d2d_{}_factor_{}' \
            '_psis_{}'.format(
                dataset, clf, non_iid, n, lr, b, alpha, radius, d2d, factor,
                '_'.join(list(map(str, psis))))
print('Saving: ', file_name)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
