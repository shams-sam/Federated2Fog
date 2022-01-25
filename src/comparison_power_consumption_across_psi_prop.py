import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


matplotlib.rcParams.update({'font.size': 42})
matplotlib.rcParams['lines.linewidth'] = 4.0
matplotlib.rcParams['lines.markersize'] = 16


fig = plt.figure(figsize=(10, 8))

# for 125 node setting: set num_workers=125, batch=480
# for 625 node setting: set num_workers=125, batch=96
dataset = 'fmnist'
clf = 'svm'
num_workers = 625
lr = 0.01
decay = 0.1
batch = 96
rounds = 2
radius = 'graph_multi'
num_layer = 4
eut_power = 250
lut_power = 10
const_time = 0.25

lut_psi_non_iid = '../history/history_{}_fcn_fog_uniform_non_iid_1_num_workers_{}_lr_0.01_batch_{}_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_0.9_dyn_True_psi_{}.pkl'
eut_psi_non_iid = '../history/history_{}_fcn_fl_uniform_non_iid_1_num_workers_{}_lr_0.01_batch_{}_repeat_1.pkl'.format(dataset, num_workers, batch)

x_ax = [100.0, 1000.0, 10000.0, 100000.0]
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


    total_lut_power = sum(num_consensus[:lut_rounds+1])*num_workers*lut_power + (num_workers/5)*eut_power*lut_rounds
    total_eut_power = num_workers*eut_power*eut_rounds

    luts.append(total_lut_power*const_time)
    euts.append(total_eut_power*const_time)

    print(eut_rounds, lut_rounds, (euts[-1]-luts[-1])/(euts[-1]))
width = 0.2
x_pos = np.arange(1, len(x_ax)+1)
plt.axhline(euts[0]/10000, label='centralized', color='k')
plt.bar(x_pos, np.array(luts)/10000, width, color='r', label='our method')


plt.legend(loc='upper right', ncol=1, bbox_to_anchor=(-0, 1.2, 1.5, .27), mode='expand', frameon=False)
plt.grid()
plt.xticks(x_pos, ['{0:.0g}'.format(_) if _ > 10 else '{0:.0f}'.format(_) for _ in x_ax], rotation=10)
plt.ylabel('energy \nconsumption ' + r'($\times 10^4$ J)')
plt.xlabel(r"parameter divergence ($\psi$)")
for format_, dpi in zip(['png', 'eps'], [100, 300]):
    figname = '../plots/comparison_eut_lut_power_{}_{}_across_psi_proposal.{}'.format(dataset, num_workers, format_)
    print('Saving...', figname)
    plt.savefig(figname, bbox_inches='tight', dpi=dpi, format=format_)
