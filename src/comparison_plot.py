import matplotlib.pyplot as plt
import pickle as pkl

clf = 'svm'
lr = 0.03
b = 1200
n = 50

files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'
colors = ['r--', 'b--', 'g--', 'k--']

range_ = list(range(1, 5))
for i in range_:
    x_ax, y_ax = pkl.load(open(files.format(clf, i, n, lr, b), 'rb'))
    plt.plot(x_ax, y_ax, colors[i-1], label=str(i))
plt.legend()
file_name = '../plots/mnist_{}_fog_uniform_non_iid_{}_{}_num_workers_{}_lr_{}_batch_{}.png'.format(
    clf, range_[0], range_[-1], n, lr, b
)
print('Saving: ', file_name)
plt.savefig(file_name)
