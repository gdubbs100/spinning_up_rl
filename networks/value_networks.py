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
        x = self.net(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self,H, W, C, intermediate_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (B, 64*7*7)
        )
        self.fc = nn.Linear(64 * H * W, 128)

    def forward(self, x):
        # x: (B, 7, 7, 3) → (B, 3, 7, 7)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.fc(x)  # (B, 128)
        return x


class CNNQnet(nn.Module):

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        intermediate_dim: int, 
        n_layers: int):

        super().__init__()
        ## assume image with 3 channels
        H, W, C = state_dim
        layers = [
            ConvLayer(H=H, W=W, C=C, intermediate_dim=intermediate_dim)
        ]
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
        x = self.net(x)
        return x