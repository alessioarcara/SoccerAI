from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric_temporal.signal import Discrete_Signal, DynamicGraphTemporalSignal

Array = Union[np.ndarray, torch.Tensor]


def get_feature_idx(name: str, feature_names: Sequence[str]) -> Optional[int]:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def make_complement(idx: int) -> Callable[[Array], None]:
    def complement_to_one(x: Array, idx=idx) -> None:
        x[:, idx] = 1.0 - x[:, idx]

    return complement_to_one


def make_signflip(idx: int) -> Callable[[Array], None]:
    def sign_flip(x: Array, idx=idx) -> None:
        x[:, idx] = -x[:, idx]

    return sign_flip


class BaseRandomFlip(BaseTransform):
    """
    Random pitch flip applied to the node features.

    The flip is decided once per call and applied to every frame of a chain,
    so that a temporal example stays consistent. Inputs are never modified
    in place: the stored dataset tensors are shared between examples (and
    between epochs), so the flipped copy is returned as a new object.
    """

    def __init__(self, p: float):
        self.p = p
        self._ops: List[Callable[[Array], None]] = []

    def _maybe(self) -> bool:
        return torch.rand(1).item() < self.p

    def _flip(self, x: Array) -> Array:
        x = x.clone() if isinstance(x, torch.Tensor) else np.array(x, copy=True)
        for op in self._ops:
            op(x)
        return x

    def forward(
        self, data: Union[Data, Discrete_Signal]
    ) -> Union[Data, Discrete_Signal]:
        if not self._maybe():
            return data

        if isinstance(data, DynamicGraphTemporalSignal):
            additional = {k: getattr(data, k) for k in data.additional_feature_keys}
            return DynamicGraphTemporalSignal(
                edge_indices=data.edge_indices,
                edge_weights=data.edge_weights,
                features=[self._flip(f) for f in data.features],
                targets=data.targets,
                **additional,
            )

        if isinstance(data, Discrete_Signal):
            raise TypeError(f"Unsupported temporal signal: {type(data).__name__}")

        data.x = self._flip(data.x)
        return data


class RandomHorizontalFlip(BaseRandomFlip):
    """Mirror the pitch along its length (x -> 1 - x)."""

    COMPLEMENT = ("x",)
    SIGN_FLIP = ("vx", "cos", "goal_cos", "dvx")

    def __init__(self, feature_names: Sequence[str], p: float):
        super().__init__(p)
        _register_ops(self, feature_names, self.COMPLEMENT, self.SIGN_FLIP)


class RandomVerticalFlip(BaseRandomFlip):
    """Mirror the pitch along its width (y -> 1 - y)."""

    COMPLEMENT = ("y",)
    SIGN_FLIP = ("vy", "sin", "goal_sin", "dvy")

    def __init__(self, feature_names: Sequence[str], p: float):
        super().__init__(p)
        _register_ops(self, feature_names, self.COMPLEMENT, self.SIGN_FLIP)


def _register_ops(
    transform: BaseRandomFlip,
    feature_names: Sequence[str],
    complement: Sequence[str],
    sign_flip: Sequence[str],
) -> None:
    for name in complement:
        if (idx := get_feature_idx(name, feature_names)) is not None:
            transform._ops.append(make_complement(idx))
    for name in sign_flip:
        if (idx := get_feature_idx(name, feature_names)) is not None:
            transform._ops.append(make_signflip(idx))
