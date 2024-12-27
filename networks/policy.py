import torch
import torch.nn as nn
import torch.distributions as dist


class DiscretePolicy(nn.Module):

    def __init__(self, n_features:int, n_actions:int, n_intermediate:int=10, n_layers:int=1):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        layers = [nn.Linear(n_features, n_intermediate), nn.ReLU()]
        for i in range(n_layers):
            layers.append(
                nn.Linear(n_intermediate, n_intermediate)
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(n_intermediate, n_actions)
        )
        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = self.net(x)
        return dist.Categorical(logits=x)

class ContinuousPolicy(nn.Module):
    ## NOTE: currently uses a Gaussian Dist

    def __init__(self, n_features:int, n_actions:int, n_intermediate:int=10, n_layers:int=1):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.logstd = nn.Parameter(torch.zeros(n_actions) - .5)
        layers = [nn.Linear(n_features, n_intermediate), nn.ReLU()]
        for i in range(n_layers):
            layers.append(
                nn.Linear(n_intermediate, n_intermediate)
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(n_intermediate, n_actions)
        )
        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = self.net(x)
        return dist.Normal(loc=x, scale = self.logstd.exp())