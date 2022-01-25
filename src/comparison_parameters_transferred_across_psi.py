import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 8

fig = plt.figure(figsize=(10, 8))

# for 125 node setting: set num_workers=125, batch=480, num_layer=3
# for 625 node setting: set num_workers=125, batch=96, num_layer=4
dataset = 'fmnist'
clf = 'svm'
num_workers = 125
lr = 0.01
decay = 0.1
batch = 480
rounds = 2
radius = 'graph_multi'
num_params = 7850
num_layer = 4
eut_power = 250
lut_power = 30

lut_psi_non_iid = '../history/history_{}_fcn_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_batch_{}_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.9_dyn_True_psi_{}.pkl'
eut_psi_non_iid = '../history/history_{}_fcn_fl_uniform_non_iid_1_num_workers_{}_lr_0.01_batch_{}_repeat_1.pkl'.format(dataset, num_workers, batch)

x_ax = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
luts = []
euts = []

for psi in x_ax:
    eut_file = eut_psi_non_iid
    lut_file = lut_psi_non_iid.format(dataset, num_workers, batch, psi)

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

    total_eut_params = num_params * (num_workers + num_workers/5 + num_workers/25) * eut_rounds
    total_lut_params = num_params * (num_workers/5 + num_workers/25) * lut_rounds
    luts.append(total_lut_params)
    euts.append(total_eut_params)

    print(eut_rounds, lut_rounds, (euts[-1]-luts[-1])/(euts[-1]))

width = 0.2
x_pos = np.arange(1, len(x_ax)+1)
plt.bar(x_pos-width/2, np.array(euts)/10000, width, color='k', label='EUT')
plt.bar(x_pos+ width/2, np.array(luts)/10000, width, color='r', label='MH-FL')

plt.legend(loc='upper right', ncol=5, bbox_to_anchor=(-0, 1.1, 1.0, .05), mode='expand', frameon=False)
plt.grid()
plt.xticks(x_pos, ['{0:.0g}'.format(_) if _ > 100 else '{0:.0f}'.format(_) for _ in x_ax], rotation=25)
plt.ylabel('accumulated number of\nparameters transferred between\nnetwork layers ' + r'($\times 10^7$)')
plt.xlabel(r"$\psi$")
for format_, dpi in zip(['png', 'eps'], [100, 300]):
    figname = '../plots/comparison_params_transferred_{}_{}_across_psi.{}'.format(dataset, num_workers, format_)
    print('Saving...', figname)
    plt.savefig(figname, bbox_inches='tight', dpi=dpi, format=format_)
