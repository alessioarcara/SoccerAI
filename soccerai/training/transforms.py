import torch
from torch_geometric.transforms import BaseTransform

# The first columns, in order, are:
# 0: "x",
# 1: "y",
# 2: "is_possession_team_1",
# 3: "is_ball_carrier_1",
# 4: "vx",
# 5: "vy",
# 6: "players_cos",
# 7: "players_sin


class BaseRandomFlip(BaseTransform):
    def __init__(self, p: float):
        self.p = p

    def _maybe(self) -> bool:
        return torch.rand(1).item() < self.p


class RandomHorizontalFlip(BaseRandomFlip):
    def forward(self, data):
        if self._maybe():
            data.x[:, 0] = 1.0 - data.x[:, 0]
            data.x[:, 4] = -data.x[:, 4]
            data.x[:, 6] = -data.x[:, 6]
        return data


class RandomVerticalFlip(BaseRandomFlip):
    def forward(self, data):
        if self._maybe():
            data.x[:, 1] = 1.0 - data.x[:, 1]
            data.x[:, 5] = -data.x[:, 5]
            data.x[:, 7] = -data.x[:, 7]
        return data
