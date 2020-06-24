import matplotlib.pyplot as plt
import pickle as pkl

clf = 'fcn'
lr = 0.03
b = 1200
n = 50
radius = 2
non_iid = 1

files = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}.pkl'
file_ = '../history/history_mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}.pkl'.format(
    clf, non_iid, n, lr, b
)

colors = ['r--', 'b--', 'g--', 'c--', 'k']

x_ax, y_ax = pkl.load(open(file_, 'rb'))
plt.plot(x_ax, y_ax, colors[-1], label='avg')
consensus = [1, 5, 10, 50]
for i in range(len(consensus)):
    x_ax, y_ax = pkl.load(open(files.format(clf, non_iid, n, lr, b, consensus[i], radius), 'rb'))
    plt.plot(x_ax, y_ax, colors[i], label=str(consensus[i]))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('{} consensus rounds comparison non-iid {}'.format(clf, non_iid))
plt.legend()
file_name = '../plots/mnist_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}.png'.format(
    clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus)))
)
print('Saving: ', file_name)
plt.savefig(file_name)
