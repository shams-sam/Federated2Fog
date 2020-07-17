from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 8


fig = plt.figure(figsize=(10, 8))

# dataset = 'mnist'
# F_0 = 0.0776
# delta = 0.05
# num_workers = 125
# lr = 0.004

dataset = 'fmnist'
F_0 = 0.0638
delta = 0.05
lr = 0.006

iid = 1
num_workers = 125
batch = 480

dataset = 'fmnist'
num_workers = 625
lr = 0.006
batch = 96
F_0 = 0.0871

xticks = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055]
eps = np.array(xticks)
k_theory = np.ceil(np.log(eps/F_0)/np.log(1-delta))
k = defaultdict(list)
deltas = [0.99]

for mul in deltas:
    x_ax, y_ax, l_test, _, _, _ = pkl.load(open('../history/history_{}_svm_fog_uniform_non_iid_{}_num_workers_{}_lr_{}_decay_0.1_batch_{}_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_4_alpha_0.9_dyn_True_delta_multiplier_{}_omega_1.1.pkl'.format(dataset, iid, num_workers, lr, batch, mul), 'rb'))
    for e in eps:
        found = 0
        for epoch, loss in enumerate(l_test):
            if loss < e:
                print(loss, e)
                k[mul].append(epoch)
                break
print(k[0.99])
width = 0.002
plt.bar(eps-width/2, k_theory, width, color='b', label='theory')
plt.bar(eps + width/2, k[0.99], width, color='r', label='simulation')


plt.legend(loc='upper right', ncol=5, bbox_to_anchor=(-0, 1.1, 1.0, .05), mode='expand', frameon=False)
plt.grid()
plt.xticks(xticks, rotation=25)
plt.xlabel("$\epsilon'$")
plt.ylabel('$\kappa$')
for format_ in ['png', 'eps']:
    fig_name = '../plots/comparison_theory_practical_delta_{}_{}.{}'.format(dataset, num_workers, format_)
    print('Saving...', fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300, format=format_)
