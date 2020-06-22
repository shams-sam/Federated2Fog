import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.reshape(-1, 784)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
