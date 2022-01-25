from math import ceil


class Arguments():
    def __init__(
            self,
            dataset='mnist',
            num_workers=125,
            num_clusters=[25, 5, 1],
            uniform_clusters=True,
            shuffle_workers=False,
            batch_size=False,
            test_batch_size=64,
            epochs=25,
            lr=0.01,
            nesterov=False,
            eta=10,
            decay=1e-1,
            no_cuda=False,
            seed=1,
            log_interval=1,
            save_model=True,
            stratify=True,
            uniform_data=False,
            shuffle_data=True,
            non_iid=1,
            repeat=1,
            rounds=2,
            radius=0.6,
            use_same_graphs=True,
            graph='multi',
            d2d=1.0,
            factor=2,
            var_theta=True,
            true_eps=True,
            alpha=1e-2,
            dynamic_alpha=False,
            alpha_multiplier=False,
            delta_or_psi = 'delta',
            psi=1e-6,
            topology='rgg',
            delta_multiplier=0.99,
            dynamic_delta=False,
            omega=1.5,
            F_0=6.250795,
            F_optim=0.753122,
            eps_multiplier=1.000001,
            kappa=15,
    ):
        # data config
        self.dataset = dataset
        self.num_trains = {'mnist': 60000, 'cifar': 50000, 'fmnist': 60000}
        self.num_train = self.num_trains[self.dataset]*repeat
        self.num_tests = {'mnist': 10000, 'cifar': 10000, 'fmnist': 10000}
        self.num_test = self.num_tests[self.dataset]
        self.input_sizes = {'mnist': 28*28, 'cifar': 3*32*32, 'fmnist': 28*28}
        self.input_size = self.input_sizes[self.dataset]
        self.output_sizes = {'mnist': 10, 'cifar': 10, 'fmnist': 10}
        self.output_size = self.output_sizes[self.dataset]
        self.stratify = save_model
        self.uniform_data = uniform_data
        self.shuffle_data = shuffle_data
        self.non_iid = non_iid
        self.repeat = repeat

        # worker clustering config
        self.num_workers = num_workers
        self.num_clusters = num_clusters
        self.uniform_clusters = uniform_clusters
        self.shuffle_workers = shuffle_workers

        # training config
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = ceil(self.num_train/self.num_workers)
        self.test_batch_size = test_batch_size
        if not self.test_batch_size:
            self.test_batch_size = self.num_test
        self.epochs = epochs
        self.lr = lr
        self.nesterov = nesterov
        self.eta = eta
        self.decay = decay
        self.no_cuda = no_cuda
        self.seed = seed

        # logging config
        self.log_interval = log_interval
        self.save_model = save_model

        # laplacian consensus
        self.rounds = rounds
        self.radius = radius
        self.use_same_graphs=use_same_graphs
        if graph == 'multi':
            self.graphs = [
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_3.2_rho_0.7500.pkl',
                'topology_rgg_degree_4.0_rho_0.3750.pkl',
            ][-len(num_clusters):]
        else:
            self.graphs='topology_rgg_degree_3.2_rho_0.7500.pkl'
        self.d2d = d2d
        self.factor = factor
        # Constant number of consensus rounds if False
        # from the parameter rounds
        self.var_theta = var_theta
        self.true_eps = true_eps
        # constant sigma value
        # set to False for calculating according to 26 or 41
        # based on dynamic alpha
        self.alpha = alpha
        # sigma calculated for every epoch
        self.dynamic_alpha = dynamic_alpha
        self.alpha_multiplier = alpha_multiplier
        self.delta_or_psi = delta_or_psi
        self.psi = psi
        # graph topology erdos renyi or rgg
        self.topology = topology
        # constant in equation 41
        self.delta_multiplier = delta_multiplier
        self.dynamic_delta = dynamic_delta
        # multiplier for true gradient estimation
        self.omega = omega
        self.eps_multiplier = eps_multiplier
        self.kappa=kappa
        self.F_0 = F_0
        self.F_optim = F_optim
