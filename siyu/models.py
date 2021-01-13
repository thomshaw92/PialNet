import torch
import torch.nn as nn
import torch.nn.functional as F

# class Model(nn.Module):
#     def __init__(self, n_layers, size):
#         super().__init__()
#         self.size = size
#         self.conv_in = torch.nn.Conv3d(1, 32, 1)
#         self.convs = nn.ModuleList([
#             torch.nn.Conv3d(32, 32, 3, padding=1) for _ in range(n_layers)
#         ])
#         self.conv_out = torch.nn.Conv3d(32, 1, 1)

#     def forward(self, x):
#         x = self.conv_in(x)
#         x = F.leaky_relu(x)
#         for l in self.convs:
#             x = l(x)
#             x = F.leaky_relu(x)
#         x = self.conv_out(x)
#         return x

class ASPP(nn.Module):
    def __init__(self, filters=32, dilation_rates=[1, 2, 5, 9]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv3d(32, 32, 3, dilation=r, padding=r) for r in dilation_rates
        ])

        self.merge = nn.Conv3d(len(dilation_rates) * 32, 32, 1)
        self.bn = nn.BatchNorm3d(32)

    def forward(self, x):
        branches = [F.leaky_relu(b(x)) for b in self.branches]

        cat = torch.cat(branches,1)
        merged = self.merge(cat)
        merged = self.bn(merged)

        x = F.leaky_relu(merged)

        return x

class Model(nn.Module):
    def __init__(self, n_layers, size):
        super().__init__()
        self.size = size
        self.conv_in = nn.Conv3d(1, 32, 1)
        self.bn_in = nn.BatchNorm3d(32)
        self.convs = nn.ModuleList([\
            torch.nn.Conv3d(32, 32, 3, padding=1) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            torch.nn.BatchNorm3d(32) for _ in range(n_layers)
        ])

        self.aspp = ASPP()
        self.conv_out = torch.nn.Conv3d(32, 1, 1)
        

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = F.leaky_relu(x)

        for l, bn in zip(self.convs, self.bns):
            x = l(x)
            x = bn(x)
            x = torch.leaky_relu(x)
        x = self.aspp(x)
        x = self.conv_out(x)
        return x