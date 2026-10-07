from typing import Callable, Optional, Tuple

import torch.nn as nn


def build_layers(
    n_layers: int,
    din: int,
    dout: int,
    conv_factory: Callable[[int, int], nn.Module],
    norm_factory: Callable[[int], nn.Module],
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    convs = nn.ModuleList()
    norms = nn.ModuleList()
    for i in range(n_layers):
        convs.append(conv_factory(din, i))
        norms.append(norm_factory(i))
        din = dout
    return convs, norms


def build_mlp(din: int, dmid: int, dout: Optional[int] = None) -> nn.Sequential:
    if dout is None:
        dout = dmid
    return nn.Sequential(nn.Linear(din, dmid), nn.ReLU(), nn.Linear(dmid, dout))
