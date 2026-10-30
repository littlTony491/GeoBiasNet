"""SBCU MLPs in the published [batch, joints, coordinates, time] layout.

Parameter names and four-dimensional normalization shapes match the GitHub
release. The feature MLP mixes XYZ coordinates; the temporal MLP mixes time.
"""

import torch
from torch import nn


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, 1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim, 1))

    def forward(self, x):
        mean = x.mean(dim=-2, keepdim=True)
        variance = (x - mean).square().mean(dim=-2, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.epsilon) * self.alpha + self.beta


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.epsilon) * self.alpha + self.beta


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x.transpose(-1, -2)).transpose(-1, -2)


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class MLPblock(nn.Module):
    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis="spatial"):
        super().__init__()
        if dim <= 0 or seq <= 0:
            raise ValueError("dim and seq must be positive")
        self.dim, self.seq = dim, seq
        self.fc0 = Spatial_FC(dim) if use_spatial_fc else Temporal_FC(seq)
        if not use_norm:
            self.norm0 = nn.Identity()
        elif layernorm_axis == "spatial":
            self.norm0 = LN(dim)
        elif layernorm_axis == "temporal":
            self.norm0 = LN_v2(seq)
        else:
            raise ValueError("layernorm_axis must be spatial or temporal")
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.zeros_(self.fc0.fc.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (self.dim, self.seq):
            raise ValueError(f"expected [B, N, {self.dim}, {self.seq}], got {tuple(x.shape)}")
        return x + self.norm0(self.fc0(x))


__all__ = ["MLPblock"]
