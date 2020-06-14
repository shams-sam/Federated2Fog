from math import ceil

class Arguments():
    def __init__(self):
        self.num_instances = 60000
        self.num_workers = 50
        self.batch_size = ceil(self.num_instances/self.num_workers)
        self.test_batch_size = 10000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = True
        self.save_init = False
        self.load_init = True

