from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from soccerai.models.typings import ResidualSumMode


def sum_residual(
    h: torch.Tensor,
    residual: Optional[torch.Tensor],
    mode: ResidualSumMode,
    layer_idx: int,
    n_layers: int,
) -> torch.Tensor:
    """
    Apply residual tensor according to sum strategy.

    - 'none' : no residual connection
    - 'every': add residual at all layers except the first (layer_idx > 0)
    - 'last' : add residual only before the final layer (layer_idx == n_layers - 1)
    """
    if residual is None or mode == "none":
        return h

    if mode == "every" and layer_idx > 0:
        return h + residual

    if mode == "last" and layer_idx == n_layers - 1:
        return h + residual

    return h


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
