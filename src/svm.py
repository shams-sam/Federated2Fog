import torch
import torch.nn as nn


class SVM(nn.Module):
    def __init__(self, n_feature=784, n_class=10):
        super(SVM, self).__init__()
        self.n_feature=n_feature
        self.n_class= n_class
        self.fc = nn.Linear(n_feature, n_class)
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        x = x.reshape(-1, self.n_feature)
        output = self.fc(x)
        return output
