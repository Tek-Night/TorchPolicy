import torch
from torch import nn

class TorchPolicyMLP(nn.Module):
    def __init__(self,
                 input_dim = 17,
                 output_dim = 6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)