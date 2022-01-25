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
dataset = 'mnist'
clf = 'svm'
num_workers = 125
lr = 0.01
decay = 0.1
batch = 480
rounds = 2
radius = 'graph_multi'
num_layer = 4
eut_power = 250
lut_power = 10
const_time = 0.25
decay = 0.1
decay = '_decay_{}'.format(decay) if decay else ''

lut_alpha_non_iid = '../history/history_{}_svm_fog_uniform_non_iid_1_num_workers_{}_lr_0.01{}_batch_{}_laplace_rounds_2_radius_graph_multi_d2d_1.0_factor_2_alpha_{}.pkl'
eut_alpha_non_iid = '../history/history_{}_svm_fl_uniform_non_iid_1_num_workers_{}_lr_0.01{}_batch_{}_repeat_1.pkl'.format(dataset, num_workers, decay, batch)


x_ax = [0.01, 0.1, 0.6, 0.9]
luts = []
euts = []

for alpha in x_ax:
    eut_file = eut_alpha_non_iid
    lut_file = lut_alpha_non_iid.format(dataset, num_workers, decay, batch, alpha)

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
plt.bar(x_pos-width/2, np.array(euts)/10000, width, color='k', label='EUT')
plt.bar(x_pos+ width/2, np.array(luts)/10000, width, color='r', label='MH-FL')


plt.legend(loc='upper right', ncol=5, bbox_to_anchor=(-0, 1.1, 1.0, .05), mode='expand', frameon=False)
plt.grid()
plt.xticks(x_pos, x_ax)
plt.ylabel('accumulated energy \nconsumption ' + r'($\times 10^4$ Joules)')
plt.xlabel(r"$\sigma'$")
for format_, dpi in zip(['png', 'eps'], [100, 300]):
    figname = '../plots/comparison_eut_lut_power_{}_{}_across_alpha.{}'.format(dataset, num_workers, format_)
    print('Saving...', figname)
    plt.savefig(figname, bbox_inches='tight', dpi=dpi, format=format_)
