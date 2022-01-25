import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 37})
matplotlib.rcParams['lines.linewidth'] = 4.0
matplotlib.rcParams['lines.markersize'] = 16


dataset = 'mnist'
clf = 'svm'
lr = 0.01
b = 480 #96
n = 125 #625
radius = 0.6
radius = 0.6 #'graph_multi'
d2d = 1.0
non_iids = [1]
consensus = [2]
epochs = 25
factor = 2

fig = plt.figure(figsize=(8, 7))
log_base = np.exp(1)
label = 2

eta = 10
mu = [1]

idx = 1
title = ['', '', 'a']
l_0 = 0
l_inf = 10000
line = 'loss'
non_iid = 1

ax = fig.add_subplot(111)
idx += 1
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
        ax.plot(x_ax, np.array(l_test)*100, colors[i], label=r'MH-FL, $\theta$ = ' + str(consensus[i]))
        if l_test[0]*100 >= l_0:
            print('here', consensus[i], i)
            l_0 = l_test[0]*100
            l_inf = l_test[-1]*100
    else:
        ax.plot(x_ax, l_test, colors[i], label=r'$\theta$ = ' + str(consensus[i]))

diff = (l_0 - l_inf)  # (6.25-0.75)*100
for m in mu:
    upper_bound = [(((eta-m)/eta)**(i-1)) * diff + l_inf for i in x_ax]
    # print(l_0, l_inf, diff)
    # print(upper_bound)
    ax.plot(x_ax, upper_bound, 'y', label='upper bound')

ax.set_xlabel('$k$')
line = 'loss ($10^{-2}$)'
ax.set_ylabel(line)
ax.grid(True)
ax.set_xlim(left=0, right=25)
# plt.title('({})'.format(title[idx]), y=-0.35)
ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(0, 1.1, 2.7, .4), mode='expand', frameon=False)
file_name = '../plots/{}_{}_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}_{}_{}_upper_bound'.format(
       dataset, clf, non_iid, n, lr, b, '_'.join(list(map(str, consensus))), radius, d2d, factor
)

print('Saving: ', file_name)
fig.subplots_adjust(wspace=0.3)
for format_ in ['png', 'eps']:
    plt.savefig(file_name + '.' + format_, bbox_inches='tight', dpi=300, format=format_)
