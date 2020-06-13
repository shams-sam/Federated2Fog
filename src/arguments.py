class Arguments():
    def __init__(self):
        self.num_instances = 60000
        self.num_workers = 20
        self.batch_size = self.num_instances/self.num_workers
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

