import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 8

fig = plt.figure(figsize=(10, 8))

dataset = 'fmnist'
clf = 'svm'
non_iid = [1, 10]
num_workers = 625
lr = 0.01
decay = 0.1
rounds = 2
radius = 'graph_multi'
num_params = 7850
num_layer = 4
eut_power = 250
lut_power = 10

num_workers = 625
lut_alpha_iid = '../history/history_{}_svm_fog_uniform_non_iid_10_num_workers_625_lr_0.01_decay_0.1_batch_96_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.1.pkl'.format(dataset)
eut_alpha_iid = '../history/history_{}_svm_fl_uniform_non_iid_10_num_workers_625_lr_0.01_decay_0.1_batch_96_repeat_1.pkl'.format(dataset)

lut_alpha_non_iid = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_625_lr_0.01_decay_0.1_batch_96_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.1.pkl'.format(dataset)
eut_alpha_non_iid = '../history/history_{}_svm_fl_uniform_non_iid_1_num_workers_625_lr_0.01_decay_0.1_batch_96_repeat_1.pkl'.format(dataset)

lut_psi_iid = '../history/history_{}_fcn_fog_uniform_non_iid_10_num_workers_625_lr_0.01_batch_96_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.9_dyn_True_psi_10000.0.pkl'.format(dataset)
eut_psi_iid = '../history/history_{}_fcn_fl_uniform_non_iid_10_num_workers_625_lr_0.01_batch_96_repeat_1.pkl'.format(dataset)

lut_psi_non_iid = '../history/history_{}_fcn_fog_uniform_non_iid_1_num_workers_625_lr_0.01_batch_96_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.9_dyn_True_psi_10000.0.pkl'.format(dataset)
eut_psi_non_iid = '../history/history_{}_fcn_fl_uniform_non_iid_1_num_workers_625_lr_0.01_batch_96_repeat_1.pkl'.format(dataset)


x_ax = [1, 2, 3, 4]
luts = []
euts = []
for lut_file, eut_file in zip(
        [lut_alpha_iid, lut_alpha_non_iid, lut_psi_iid, lut_psi_non_iid],
        [eut_alpha_iid, eut_alpha_non_iid, eut_psi_iid, eut_psi_non_iid]
):
    x_ax_eut, y_ax_eut, _, _ = pkl.load(open(eut_file, 'rb'))
    x_ax_lut, y_ax_lut, _, _, rounds_tr, _ = pkl.load(open(lut_file, 'rb'))
    y_ax_eut = y_ax_eut[:25]
    y_ax_lut = y_ax_lut[:25]

    layer_rounds = [[] for _ in range(num_layer)]
    for r in rounds_tr:
        for r_idx in range(len(r)):
            layer_rounds[r_idx].append(sum(r[r_idx])/len(r[r_idx]))
    num_consensus = np.array(layer_rounds[0])

    threshold = 0.98*max(y_ax_eut)
    eut_rounds= 0
    for _ in y_ax_eut:
        if _ < threshold:
            eut_rounds += 1
        else:
            break
    lut_rounds = 0
    for _ in y_ax_lut:
        if _ < threshold:
            lut_rounds += 1

    total_eut_params = num_params * (num_workers + num_workers/5 + num_workers/25 + num_workers/125) * eut_rounds
    total_lut_params = num_params * (num_workers/5 + num_workers/25 + num_workers/125) * lut_rounds
    print(total_eut_params, total_lut_params)
    print(eut_rounds, lut_rounds)
    luts.append(total_lut_params)
    euts.append(total_eut_params)

width = 0.2
x_ax = np.array(x_ax)
plt.bar(x_ax-width/2, np.array(euts)/10**7, width, color='k', label='EUT')
plt.bar(x_ax+ width/2, np.array(luts)/(10**7), width, color='r', label='MH-MT')

plt.legend(loc='upper right', ncol=5, bbox_to_anchor=(-0, 1.1, 1.0, .05), mode='expand', frameon=False)
plt.grid()
plt.xticks(x_ax, ['scenario 1', 'scenario 2', 'scenario 3', 'scenario 4'], rotation=25)
plt.ylabel('accumulated number of\nparameters transferred between\nnetwork layers ' + r'($\times 10^7$)')
for format_ in ['png', 'eps']:
    figname = '../plots/comparison_params_transferred_{}_{}.{}'.format(dataset, num_workers, format_)
    print('Saving...', figname)
    plt.savefig(figname, bbox_inches='tight', dpi=300, format=format_)
