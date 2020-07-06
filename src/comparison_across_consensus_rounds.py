import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl


matplotlib.rcParams.update({'font.size': 14})

clf = 'svm'
lr = 0.003
b = 300
n = 200
radius = 0.2
d2d = 1.0
non_iids = [1]
consensus = [1, 2, 5, 10, 15, 30]
epochs = 50
factor = 8

# clf = 'svm'
# lr = 0.01
# b = 100
# n = 600
# radius = 0.7
# d2d = 1.0
# non_iids = [1]
# consensus = [1, 3, 5, 10, 15, 30]
# epochs = 50
# factor = 8

fig = plt.figure(figsize=(20, 6))

idx = 1
title = ['', '', 'a', 'b']
for line in ['accuracy', 'loss']:
    for non_iid in non_iids:
        ax = fig.add_subplot(120 + idx)
        idx += 1
        files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}.pkl'
        file_ = '../history/history_mnist_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'.format(
            clf, non_iid, n, lr, b
        )

        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            ax.plot(x_ax, y_ax, colors[-1], label='EUT fogL')
        else:
            ax.plot(x_ax, l_test, colors[-1], label='EUT fogL')

        for i in range(len(consensus)):
            x_ax, y_ax, l_test, grad_tr = pkl.load(open(files.format(clf, non_iid, n, lr, b, consensus[i], radius, d2d, factor), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                plt.plot(x_ax, y_ax, colors[i], label='num cons = '+ str(consensus[i]))
            else:
                plt.plot(x_ax, l_test, colors[i], label='num cons = ' + str(consensus[i]))
        ax.set_xlabel('epochs')
        ax.set_ylabel(line)
        ax.grid(True)
        ax.set_xlim(left=0, right=50)
        plt.title('({})'.format(title[idx]), y=-0.22)
        if idx == 3:
            ax.legend()
file_name = '../plots/_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}'.format(
       clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus))), radius, d2d, factor
)
print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.2)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
