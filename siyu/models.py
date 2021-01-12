import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_layers, size):
        super().__init__()
        self.size = size
        self.conv_in = torch.nn.Conv3d(1, 32, 1)
        self.convs = nn.ModuleList([
            torch.nn.Conv3d(32, 32, 3, padding=1) for _ in range(n_layers)
        ])
        self.conv_out = torch.nn.Conv3d(32, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = F.leaky_relu(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
        x = self.conv_out(x)
        return x
