import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        x = self.fc1(x.reshape(-1, self.input_size))
        return F.log_softmax(x, dim=1)
