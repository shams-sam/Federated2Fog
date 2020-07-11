from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(6, 6))

F_0 = 0.0776
delta = 0.05

eps = np.array([0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055])
k_theory = np.ceil(np.log(eps/F_0)/np.log(1-delta))
k = defaultdict(list)
deltas = [0.99, 0.8]

for mul in deltas:
    x_ax, y_ax, l_test, _, _, _ = pkl.load(open('../history/history_mnist_svm_fog_uniform_non_iid_1_num_workers_125_lr_0.004_decay_0.1_batch_480_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_4_alpha_0.9_dyn_True_delta_multiplier_{}_omega_1.1.pkl'.format(mul), 'rb'))
    for e in eps:
        found = 0
        for epoch, loss in enumerate(l_test):
            if loss < e:
                k[mul].append(epoch)
                break

width = 0.002
plt.bar(eps-width/2, k_theory, width, color='b', label='thoeretical')
plt.bar(eps + width/2, k[0.99], width, color='r', label='simulation')


plt.legend()
plt.grid()
plt.xlabel('$\epsilon$')
plt.ylabel('$\kappa$')
for format_ in ['png', 'eps']:
    plt.savefig('../plots/comparison_theory_practical_delta.{}'.format(format_), bbox_inches='tight', dpi=300, format=format_)
