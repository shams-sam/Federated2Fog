from arguments import Arguments
from fcn import FCN
from distributor import get_fog_graph
from train import fog_train as train, test
import os
import pickle as pkl
import syft as sy
import torch
from torchvision import datasets, transforms


# Setups
args = Arguments()
hook = sy.TorchHook(torch)
USE_CUDA = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if USE_CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
kwargs = {}

for non_iid in range(1, 5):
    ckpt_path = '../ckpts'
    dataset = 'mnist'
    clf_type = 'fcn'
    paradigm = 'fog_uniform_non_iid_{}_num_workers_{}_lr_{}_batch_{}_laplace_{}_{}'.format(
        non_iid,
        args.num_workers,
        args.lr,
        args.batch_size,
        args.rounds,
        args.radius,
    )
    model_name = '{}_{}_{}'.format(dataset, clf_type, paradigm)
    print('+'*80)
    print(model_name)
    print('+'*80)
    
    init_path = '../init/mnist_fcn.init'
    best_path = os.path.join(ckpt_path, model_name + '.best')
    stop_path = os.path.join(ckpt_path, model_name + '.stop')

    # prepare graph and data
    fog_graph, workers = get_fog_graph(hook, args.num_workers,
                                       args.num_clusters,
                                       args.shuffle_workers,
                                       args.uniform_clusters)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    X_trains, y_trains = pkl.load(
        open('../ckpts/data_non_iid_{}_num_workers_{}_stratify_True_uniform_True.pkl'.format(
            non_iid, args.num_workers), 'rb'))

    print(fog_graph)

    best = 0

    # Fire the engines
    model = FCN().to(device)
    model.load_state_dict(torch.load(init_path))
    print('Load init: {}'.format(init_path))

    best = 0

    x_ax = []
    y_ax = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, fog_graph, workers, X_trains, y_trains,
              device, epoch, 'nll', 'laplacian',
              args.rounds, args.radius, args.d)
        acc = test(args, model, device, test_loader, best, epoch)
        y_ax.append(acc)
        x_ax.append(epoch)

        if args.save_model and acc > best:
            best = acc
            torch.save(model.state_dict(), best_path)
            print('Model best  @ {}, acc {}: {}\n'.format(
                epoch, acc, best_path))

    if (args.save_model):
        torch.save(model.state_dict(), stop_path)
        print('Model stop: {}'.format(stop_path))

    hist_file = '../history/history_{}.pkl'
    pkl.dump((x_ax, y_ax), open(hist_file.format(model_name), 'wb'))
    print('Saved: ', hist_file)

    import matplotlib.pyplot as plt
    plt.plot(x_ax, y_ax)
    plot_file = '../plots/{}.png'
    plt.savefig(plot_file.format(model_name))
    print('Saved: ', plot_file)
