import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 14})

# clf = 'svm'
# lr = 0.01
# b = 300
# n = 200
# radius = 0.2
# d2d = 1.0
# non_iids = [1]
# alpha = [5, 50, 500, 5000, 50000]
# epochs = 50
# factor = 8
# rounds = 2
# radius = 0.7
# d2d = 1.0
# factor = 16
# num_layer = 4

clf = 'svm'
lr = 0.01
b = 32
n = 125
radius = 1.0
d2d = 1.0
non_iids = [1]
alpha = [1e-4, 1e-2, 1e-1, 3e-1, 6e-1, 8e-1]
epochs = 50
factor = 4
rounds = 2
# radius = 1.0
# radius = [0.6, 0.7, 0.9]
radius = 0.6
d2d = 1.0
num_layer = 3


rows = 3
cols = 3
fig = plt.figure(figsize=(10*cols, 6*rows))
plt_start = rows * 100 + cols * 10
idx = 1
title = ['', '', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
for line in ['accuracy', 'loss', 'rounds'] + \
    ['layer_{}'.format(_) for _ in range(num_layer)]:
    for non_iid in non_iids:
        plt_num = plt_start + idx
        ax = fig.add_subplot(plt_num)
        idx += 1
        files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}_alpha_{}.pkl'
        file_ = '../history/history_mnist_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_repeat_1.pkl'.format(
            clf, non_iid, n, lr, b
        )

        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, colors[-1], label='EUT fogL')
        else:
            ax.plot(x_ax, l_test, colors[-1], label='EUT fogL')

        for i in range(len(alpha)):
            x_ax, y_ax, l_test, grad_tr, rounds_tr, _ = pkl.load(open(files.format(clf, non_iid, n, lr,
 b, rounds, radius, d2d, factor, alpha[i]), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                ax.plot(x_ax, y_ax, colors[i], label='sigma = '+ str(alpha[i]))
            elif line == 'loss':
                ax.plot(x_ax, l_test, colors[i], label='sigma = ' + str(alpha[i]))
            elif line == 'rounds':
                avg_rounds = []
                for r in rounds_tr:
                    r = np.array([_ for layer in r for _ in layer])
                    avg_rounds.append(r.sum()/len(r))
                ax.plot(x_ax, avg_rounds[:epochs], colors[i], label='sigma = ' + str(alpha[i]))
            elif 'layer' in  line:
                l_num = int(line.split('_')[-1])
                layer_rounds = [[] for _ in range(num_layer)]
                for r in rounds_tr:
                    for r_idx in range(len(r)):
                        layer_rounds[r_idx].append(sum(r[r_idx])/len(r[r_idx]))

                ax.plot(x_ax, layer_rounds[l_num][:epochs], colors[i], label=line)
                        

        ax.set_xlabel('epochs')
        ax.set_ylabel(line)
        ax.grid(True)
        # ax.set_xlim(left=0, right=50)
        ax.set_title('({})'.format(title[idx]), y=-0.22)
        if idx == 4:
            ax.legend()
file_name = '../plots/_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}_alpha'.format(
       clf, non_iid, n, lr, b, '_'.join(list(map(str, alpha))), radius, d2d, factor
)
print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.3, hspace=0.4)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
