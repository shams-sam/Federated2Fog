import matplotlib.pyplot as plt
import pickle as pkl

clf = 'svm'
lr = 0.006
b = 300
n = 200
radius = 0.2
d2d = 1.0
non_iids = [1]
consensus = [1, 5, 10, 15]
epochs = 50


for line in ['accuracy', 'loss']:
    for non_iid in non_iids:
        files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}.pkl'
        file_ = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'.format(
            clf, non_iid, n, lr, b
        )

        plt.figure()

        colors = ['r--', 'b--', 'g--', 'c--', 'k']

        x_ax, y_ax, l_test = pkl.load(open(file_, 'rb'))
        x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
        if line=='accuracy':
            plt.plot(x_ax, y_ax, colors[-1], label='avg')
        else:
            plt.plot(x_ax, l_test, colors[-1], label='avg')

        for i in range(len(consensus)):
            x_ax, y_ax, l_test = pkl.load(open(files.format(clf, non_iid, n, lr, b, consensus[i], radius, d2d), 'rb'))
            x_ax, y_ax, l_test = x_ax[:epochs], y_ax[:epochs], l_test[:epochs]
            if line == 'accuracy':
                plt.plot(x_ax, y_ax, colors[i], label=str(consensus[i]))
            else:
                plt.plot(x_ax, l_test, colors[i], label=str(consensus[i]))
        plt.xlabel('epochs')
        plt.ylabel(line)
        plt.title('{} consensus rounds comparison non-iid {}'.format(clf, non_iid))
        plt.legend()
        file_name = '../plots/mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}.png'.format(
            clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus))), radius, d2d, line
        )
        print('Saving: ', file_name)
        plt.savefig(file_name)
