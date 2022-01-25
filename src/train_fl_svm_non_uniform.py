from arguments import Arguments
from svm import SVM
from distributor import get_fog_graph
from train import fl_train as train, test
import os
import pickle as pkl
import syft as sy
import sys
import torch
from utils import get_testloader


# Setups
args = Arguments()
hook = sy.TorchHook(torch)
USE_CUDA = not args.no_cuda and torch.cuda.is_available()

simple_avg = True

torch.manual_seed(args.seed)
device = torch.device("cuda" if USE_CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
kwargs = {}

for non_iid in range(1, 2):
    non_iid = args.non_iid
    ckpt_path = '../ckpts'
    dataset = args.dataset
    clf_type = 'svm'
    paradigm = 'fl_uniform_False_non_iid_{}_num_workers_{}_lr_{}_decay_{}_batch_{}_repeat_{}_avg_{}'.format(
        non_iid,
        args.num_workers,
        args.lr,
        args.decay,
        args.batch_size,
        args.repeat,
        simple_avg,
    )
    model_name = '{}_{}_{}'.format(dataset, clf_type, paradigm)
    file_ = '../logs/{}.log'.format(model_name)
    print("Logging: ", file_)
    log_file = open(file_, 'w')
    std_out = sys.stdout
    sys.stdout = log_file
    print('+'*80)
    print(model_name)
    print('+'*80)
    
    init_path = '../init/{}_svm.init'.format(dataset)
    best_path = os.path.join(ckpt_path, model_name + '.best')
    stop_path = os.path.join(ckpt_path, model_name + '.stop')

    # prepare graph and data
    fog_graph, workers = get_fog_graph(hook, args.num_workers,
                                       args.num_clusters,
                                       args.shuffle_workers,
                                       args.uniform_clusters, fog=False)

    test_loader = get_testloader(args)

    if non_iid == 10:
        data_file = '../ckpts/data_{}_iid_num_workers_{}' \
                    '_stratify_True_uniform_False_repeat_{}.pkl'.format(
                        dataset, args.num_workers, args.repeat)
    else:
        data_file = '../ckpts/data_{}_non_iid_{}_num_workers_{}' \
                    '_stratify_True_uniform_False_repeat_{}.pkl'.format(
                        dataset, non_iid, args.num_workers, args.repeat)
    print('Loading data: {}'.format(data_file))
    X_trains, y_trains = pkl.load(
        open(data_file, 'rb'))

    print(fog_graph)

    best = 0

    # Fire the engines
    model = SVM(args.input_size, args.output_size).to(device)
    model.load_state_dict(torch.load(init_path))
    print('Load init: {}'.format(init_path))

    best = 0
    x_ax = []
    y_ax = []
    l_test = []
    grad_tr = []
    for epoch in range(1, args.epochs + 1):
        grad = train(args, model, fog_graph, workers, X_trains, y_trains,
                     device, epoch, loss_fn='hinge', simple_avg=simple_avg)
        acc, loss = test(args, model, device, test_loader, best, epoch,
                         loss_fn='hinge')
        y_ax.append(acc)
        x_ax.append(epoch)
        l_test.append(loss)
        grad_tr.append(grad)

        if args.save_model and acc > best:
            best = acc
            torch.save(model.state_dict(), best_path)
            print('Model best  @ {}, acc {}: {}\n'.format(
                epoch, acc, best_path))

    if (args.save_model):
        torch.save(model.state_dict(), stop_path)
        print('Model stop: {}'.format(stop_path))

    hist_file = '../history/history_{}.pkl'.format(model_name)
    pkl.dump((x_ax, y_ax, l_test, grad_tr), open(hist_file, 'wb'))
    print('Saved: ', hist_file)

    import matplotlib.pyplot as plt
    plt.plot(x_ax, y_ax)
    plot_file = '../plots/{}.png'.format(model_name)
    plt.savefig(plot_file)
    print('Saved: ', plot_file)

    log_file.close()
    sys.stdout = std_out
