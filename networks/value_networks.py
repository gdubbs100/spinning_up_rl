import torch
import torch.nn as nn

class Qnet(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        intermediate_dim: int, 
        n_layers: int):

        super().__init__()
        layers = [
            nn.Linear(state_dim, intermediate_dim), 
            nn.ReLU()]
        for i in range(n_layers):
            layers.append(
                nn.Linear(intermediate_dim,intermediate_dim)
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(intermediate_dim, action_dim)
        )
        self.net = nn.Sequential(
            *layers
        )
    
    def forward(self, x):
        # x = torch.tensor(x)
        x = self.net(x)
        return x