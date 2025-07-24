import torch
from torch import nn


class MEGCompose(nn.Module):
    def __init__(self, transforms):
        super(MEGCompose, self).__init__()
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01, p=0.5):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std
            x = x + noise
        return x


class UniformScaling(nn.Module):
    def __init__(self, scale_min=0.9, scale_max=1.1, p=0.5):
        super(UniformScaling, self).__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            scale = torch.empty((x.size(0), x.size(1), 1), device=x.device).uniform_(self.scale_min, self.scale_max)
            x = x * scale
        return x


class ChannelDropout(nn.Module):
    def __init__(self, dropout_prob=0.1, p=0.3):
        super(ChannelDropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            mask = torch.rand((x.size(0), x.size(1), 1), device=x.device) > self.dropout_prob
            x = x * mask
        return x