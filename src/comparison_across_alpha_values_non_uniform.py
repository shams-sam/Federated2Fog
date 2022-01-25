import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# histories
# non-uniform EUT: ../history/history_mnist_svm_fl_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_repeat_1.pkl
# non-uniform alpha = 0.1: ../history/history_mnist_svm_fog_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.1.pkl


matplotlib.rcParams.update({'font.size': 39})
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
fig = plt.figure(figsize=(10*cols, 7*rows))
plt_start = rows * 100 + cols * 10
idx = 1

histories = [
    # '../history/history_mnist_svm_fl_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_repeat_1_avg_True.pkl',
    # '../history/history_mnist_svm_fl_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_repeat_1_avg_False.pkl',
    '../history/history_mnist_svm_fog_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.01_avg_True.pkl', 
    '../history/history_mnist_svm_fog_uniform_False_non_iid_1_num_workers_125_lr_0.01_decay_0.1_batch_480_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.01_avg_False.pkl'
]

labels =[
    # 'EUT,uniform', 'EUT,non-uniform',
    r'$\theta$=15,uniform', r'$\theta$=15,non-uniform'
]

title = ['', '', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
for line in ['accuracy', 'loss']:
    for non_iid in non_iids:
        plt_num = plt_start + idx
        ax = fig.add_subplot(plt_num)
        idx += 1
        colors = ['r.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        for i, fname in enumerate(histories):
            if 'EUT' not in labels[i]:
                x_ax, y_ax, l_test, grad_tr, rounds_tr, _ = pkl.load(open(fname, 'rb'))
            else:
                x_ax, y_ax, l_test, grad_tr = pkl.load(open(fname, 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                ax.plot(x_ax, y_ax, colors[i], label=labels[i])
            elif line == 'loss':
                ax.plot(x_ax, np.array(l_test)*100, colors[i], label=labels[i])
        ax.set_xlabel('$k$')
        if line == 'loss':
            line = r'loss ($\times 10^{-2}$)'
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=25)
        ax.set_title('({})'.format(title[idx]), y=-0.35)
        if idx == 3:
            ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(-1.425, 1.15, 2.5, .1), mode='expand', frameon=False)
file_name = '../plots/{}_{}_fog_uniform_False_non_iid_{}_num_workers_{}_lr_{}{}_batch_{}_laplace_alphas_{}_radius_{}_d2d_{}_factor_{}_alpha'.format(
       dataset, clf, non_iid, n, lr, decay, b, '_'.join(list(map(str, alpha))), radius, d2d, factor
)
print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.35, hspace=0.5)
for format_, dpi in zip(['png', 'eps'], [300, 300]):
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=dpi, format=format_)
