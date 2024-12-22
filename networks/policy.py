import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

import gymnasium as gym

## TODO: differentiate between discrete and continuous policies?
class Policy(nn.Module):

    def __init__(self, n_features, n_actions, n_intermediate=10):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(n_features, n_intermediate),
            nn.ReLU(),
            nn.Linear(n_intermediate, n_actions)
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = self.net(x)
        return x