from fcn import FCN
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from svm import SVM

model = SVM()
params = model.state_dict().keys()
clf = 'svm'
lr = 0.01
b = 300
n = 200
radius = 0.2
d2d = 1.0
non_iids = [1]
consensus = [1, 3, 5, 10, 15, 50]
epochs = 25
factor = 16
D = 60000


for line in params:
    for non_iid in non_iids:
        files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_rounds_{}_radius_{}_d2d_{}_factor_{}.pkl'
        fl = '../history/history_mnist_{}_fl_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'.format(
            clf, non_iid, n, lr, b
        )
        file_ = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'.format(
            clf, non_iid, n, lr, b
        )

        plt.figure()

        colors = ['r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'k.:', 'k']

        x_ax, y_ax, l_test, grad_tr = pkl.load(open(fl, 'rb'))
        grad_tr = np.array([_[line] for _ in grad_tr])
        x_ax, grad_tr = x_ax[:epochs], grad_tr[:epochs]
        plt.plot(x_ax, grad_tr, colors[-1], label='FL')

        for i in range(3, len(consensus)):
            x_ax, y_ax, l_test, grad_tr = pkl.load(open(files.format(clf, non_iid, n, lr, b, consensus[i], radius, d2d, factor), 'rb'))
            grad_tr = np.array([_[line] for _ in grad_tr])
            x_ax, grad_tr = x_ax[:epochs], grad_tr[:epochs]
            plt.plot(x_ax, grad_tr, colors[i], label=str(consensus[i]))
        plt.xlabel('epochs')
        plt.ylabel('gradient')
        plt.title('{} grad {} rounds comparison non-iid {}'.format(clf, line, non_iid))
        plt.legend()
        file_name = '../plots/mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}_{}.png'.format(
            clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus))), radius, d2d, line, factor
        )
        print('Saving: ', file_name)
        plt.savefig(file_name)
