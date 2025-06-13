import torch
from torch_geometric.transforms import BaseTransform


class BaseRandomFlip(BaseTransform):
    def __init__(self, p: float):
        self.p = p

    def _maybe(self) -> bool:
        return torch.rand(1).item() < self.p


class RandomHorizontalFlip(BaseRandomFlip):
    def forward(self, data):
        if self._maybe():
            data.x[:, 0] = 1.0 - data.x[:, 0]
            data.x[:, 5] = -data.x[:, 5]
        return data


class RandomVerticalFlip(BaseRandomFlip):
    def forward(self, data):
        if self._maybe():
            data.x[:, 1] = 1.0 - data.x[:, 1]
            data.x[:, 4] = -data.x[:, 4]
        return data
