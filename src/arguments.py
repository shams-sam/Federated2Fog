from math import ceil


class Arguments():
    def __init__(
            self,
            num_train=60000,
            num_test=10000,
            num_workers=200,
            # for more gap among consensus
            # more workers
            # higher degree
            num_clusters=[10, 5, 2, 1],
            uniform_clusters=True,
            shuffle_workers=False,
            batch_size=False,
            test_batch_size=300,
            epochs=50,
            lr=0.006,
            no_cuda=False,
            seed=1,
            log_interval=1,
            save_model=True,
            stratify=True,
            uniform_data=True,
            shuffle_data=True,
            non_iid=1,
            rounds=1,
            radius=0.2,
            d2d=1.0,
    ):
        # data config
        self.num_train = num_train
        self.num_test = num_test
        self.stratify = save_model
        self.uniform_data = uniform_data
        self.shuffle_data = shuffle_data
        self.non_iid = non_iid

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
        self.no_cuda = no_cuda
        self.seed = seed

        # logging config
        self.log_interval = log_interval
        self.save_model = save_model

        # laplacian consensus
        self.rounds = rounds
        self.radius = radius
        self.d2d = d2d
