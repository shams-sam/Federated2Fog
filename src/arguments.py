from math import ceil


class Arguments():
    def __init__(
            self,
            num_train=60000,
            num_test=10000,
            # num_workers=200,
            # num_clusters=[10, 5, 2, 1],
            # num_workers=500,
            # num_clusters=[100, 20, 4, 1],
            num_workers=125,
            num_clusters=[25, 5, 1],
            # for more gap among consensus
            # more workers
            # higher degree
            uniform_clusters=True,
            shuffle_workers=False,
            batch_size=False,
            test_batch_size=64,
            epochs=50,
            lr=0.01,
            nesterov=False,
            eta=10,
            decay=1e-1,
            no_cuda=False,
            seed=1,
            log_interval=1,
            save_model=True,
            stratify=True,
            uniform_data=True,
            shuffle_data=True,
            non_iid=1,
            repeat=1,
            rounds=2,
            radius=0.6,
            use_same_graphs=True,
            # graphs='topology_rgg_degree_3.2_rho_0.7500.pkl',
            graphs=[
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_3.2_rho_0.7500.pkl',
                'topology_rgg_degree_4.0_rho_0.3750.pkl',
            ],
            # radius=[0.6, 0.7, 0.9],
            d2d=1.0,
            factor=4,
            var_theta=True,
            true_eps=False,
            alpha=9e-1,
            dynamic_alpha=True,
            # 1 from 2.5 to 2
            alpha_multiplier=[2e6, 1e12, 1e2],
            topology='rgg',
            delta_multiplier=0.99,
            dynamic_delta=False,
            omega=1.1,
            F_0=0.0776,
            F_optim=0,
            eps_multiplier=1.0001,
            kappa=1,
    ):
        # data config
        self.num_train = num_train*repeat
        self.num_test = num_test
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
        self.eta=eta
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
        self.graphs=graphs
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
