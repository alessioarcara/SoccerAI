from typing import Callable, List, Optional, Sequence

import torch
from torch_geometric.transforms import BaseTransform


def get_feature_idx(name: str, feature_names: Sequence[str]) -> Optional[int]:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def make_complement(idx: int):
    def complement_to_one(data, idx=idx):
        data.x[:, idx] = 1.0 - data.x[:, idx]

    return complement_to_one


def make_signflip(idx: int):
    def sign_flip(data, idx=idx):
        data.x[:, idx] = -data.x[:, idx]

    return sign_flip


class BaseRandomFlip(BaseTransform):
    def __init__(self, p: float):
        self.p = p

    def _maybe(self) -> bool:
        return torch.rand(1).item() < self.p


class RandomHorizontalFlip(BaseRandomFlip):
    def __init__(self, feature_names: Sequence[str], p: float):
        super().__init__(p)

        self._ops: List[Callable] = []

        if x_idx := get_feature_idx("x", feature_names):
            self._ops.append(make_complement(x_idx))
        if vx_idx := get_feature_idx("vx", feature_names):
            self._ops.append(make_signflip(vx_idx))
        if player_cos_idx := get_feature_idx("cos", feature_names):
            self._ops.append(make_signflip(player_cos_idx))
        if goal_cos_idx := get_feature_idx("goal_cos", feature_names):
            self._ops.append(make_signflip(goal_cos_idx))

    def forward(self, data):
        if self._maybe():
            for op in self._ops:
                op(data)

        return data


class RandomVerticalFlip(BaseRandomFlip):
    def __init__(self, feature_names: Sequence[str], p: float):
        super().__init__(p)

        self._ops: List[Callable] = []

        if y_idx := get_feature_idx("y", feature_names):
            self._ops.append(make_complement(y_idx))
        if vy_idx := get_feature_idx("vy", feature_names):
            self._ops.append(make_signflip(vy_idx))
        if player_sin_idx := get_feature_idx("sin", feature_names):
            self._ops.append(make_signflip(player_sin_idx))
        if goal_sin_idx := get_feature_idx("goal_sin", feature_names):
            self._ops.append(make_signflip(goal_sin_idx))

    def forward(self, data):
        if self._maybe():
            for op in self._ops:
                op(data)

        return data
