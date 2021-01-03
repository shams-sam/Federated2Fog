import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 39})
matplotlib.rcParams['lines.linewidth'] = 4.0
matplotlib.rcParams['lines.markersize'] = 16


dataset = 'fmnist'
clf = 'svm'
lr = 0.01
b = 96
n = 625
radius = 'graph_multi'
d2d = 1.0
non_iids = [1]
consensus = [1, 2, 5, 15, 30]
epochs = 25
factor = 2

fig = plt.figure(figsize=(30, 7))
log_base = np.exp(1)
label = 2

idx = 1
title = ['', '', 'a', 'b', 'c']
for line in ['accuracy', 'loss', 'loss (log scale)']:
    for non_iid in non_iids:
        ax = fig.add_subplot(130 + idx)
        idx += 1
        # files = '../history/history_{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_nest_False_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}_repeat_1.pkl'
        # file_ = '../history/history_{}_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_decay_0.1_batch_{}_repeat_1.pkl'.format(
        #     dataset, clf, non_iid, n, lr, b
        # )
        files = '../history/history_{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_nest_False_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}_repeat_1.pkl'
        file_ = '../history/history_{}_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_decay_0.1_batch_{}_repeat_1.pkl'.format(
            dataset, clf, non_iid, n, lr, b
        )


        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, colors[-1], label='EUT')
        elif line=='loss':
            ax.plot(x_ax, np.array(l_test)*100, colors[-1], label='EUT')
        else:
            ax.plot(x_ax, l_test, colors[-1], label='EUT')
            ax.set_yscale('log', basey=log_base)
            ax.set_xscale('log', basex=log_base)

        for i in range(len(consensus)):
            x_ax, y_ax, l_test, grad_tr = pkl.load(
                open(files.format(
                    dataset, clf, non_iid, n, lr, b,
                    consensus[i], radius, d2d, factor), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                ax.plot(x_ax, y_ax, colors[i], label=r'$\theta$ = '+ str(consensus[i]))
            elif line=='loss':
                ax.plot(x_ax, np.array(l_test)*100, colors[i], label=r'$\theta$ = ' + str(consensus[i]))
            else:
                ax.plot(x_ax, l_test, colors[i], label=r'$\theta$ = ' + str(consensus[i]))
                ax.set_yscale('log', basey=log_base)
                ax.set_xscale('log', basex=log_base)
                ax.set_yticks([
                #     log_base**(-8), log_base**(-7), log_base**(-6),
                    log_base**(-4), log_base**(-3)])
                ax.set_yticklabels([
                #     '$e^{-8}$', '$e^{-7}$', '$e^{-6}$',
                    '$e^{-4}$', '$e^{-3}$'])
                ax.set_xticks([log_base**0, log_base**1, log_base**2, log_base**3, log_base**4, log_base**5])
                ax.set_xticklabels(['$e^0$', '$e^1$', '$e^2$', '$e^3$', '$e^4$', '$e^5$'])
        ax.set_xlabel('$k$')
        if line == 'loss':
            line = 'loss ($10^{-2}$)'
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=25)
        plt.title('({})'.format(title[idx]), y=-0.35)
        if idx == 3:
            ax.legend(loc='upper right', ncol=6, bbox_to_anchor=(-1.35, 1.1, 3.7, .1), mode='expand', frameon=False)
file_name = '../plots/{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}'.format(
       dataset, clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus))), radius, d2d, factor
)

print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.3)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
